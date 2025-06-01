"""
@file purpose: Defines MultiActionExecutor class for executing sequences of browser actions
This file encapsulates the logic for executing multiple browser actions with element relocation
and page stability checks, extracted from the original Agent.multi_act method.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel

from browser_use.agent.views import ActionModel as AgentActionModel, ActionResult as AgentActionResult
from browser_use.browser.session import BrowserSession
from browser_use.controller.service import Controller
from browser_use.dom.history_tree_processor.service import HistoryTreeProcessor
from browser_use.dom.views import DOMElementNode
from browser_use.utils import time_execution_async

logger = logging.getLogger(__name__)


def get_effective_element_label(node: Optional[DOMElementNode]) -> str:
	"""
	Get effective label for an element, considering various accessibility attributes.
	
	This function was moved from Agent._get_effective_label to be reusable.
	"""
	if not node:
		return ''

	# Check aria-label first
	if 'aria-label' in node.attributes:
		aria_label = node.attributes['aria-label'].strip()
		if aria_label:
			return aria_label

	# Check aria-labelledby
	if 'aria-labelledby' in node.attributes:
		labelledby_ids = node.attributes['aria-labelledby'].strip().split()
		if labelledby_ids:
			# Try to find the referenced element(s)
			# Note: This is a simplified implementation - in practice, you'd need to traverse the DOM
			# to find elements with these IDs
			pass

	# Check for label element pointing to this element
	if 'id' in node.attributes:
		element_id = node.attributes['id']
		# In a full implementation, you'd search for label[for="element_id"]
		# For now, we'll skip this complex traversal
		pass

	# Check title attribute
	if 'title' in node.attributes:
		title = node.attributes['title'].strip()
		if title:
			return title

	# Check placeholder for form elements
	if 'placeholder' in node.attributes:
		placeholder = node.attributes['placeholder'].strip()
		if placeholder:
			return placeholder

	# Check alt attribute for images
	if 'alt' in node.attributes:
		alt = node.attributes['alt'].strip()
		if alt:
			return alt

	# Check value attribute for buttons/inputs
	if 'value' in node.attributes:
		value = node.attributes['value'].strip()
		if value:
			return value

	# Fall back to text content
	text_content = node.get_all_text_till_next_clickable_element(max_depth=2)
	if text_content.strip():
		return text_content.strip()

	# Last resort: use tag name and attributes
	tag_info = node.tag_name
	if 'type' in node.attributes:
		tag_info += f"[type={node.attributes['type']}]"
	if 'name' in node.attributes:
		tag_info += f"[name={node.attributes['name']}]"
	if 'class' in node.attributes:
		classes = node.attributes['class'][:50]  # Truncate long class lists
		tag_info += f"[class={classes}]"
	
	return tag_info


class MultiActionExecutor:
	"""
	Executes a sequence of browser actions with element relocation and page stability checks.
	
	This class encapsulates the logic that was previously in Agent.multi_act, providing
	better separation of concerns and reusability for different contexts (e.g., MCP servers).
	"""

	def __init__(
		self,
		browser_session: BrowserSession,
		controller: Controller,
		action_model_class: Type[AgentActionModel],
		page_extraction_llm: Optional[BaseChatModel],
		sensitive_data: Optional[Dict[str, str]],
		available_file_paths: Optional[List[str]],
		generic_controller_context: Optional[Any],
		unexpected_elements_threshold: int,
		wait_between_actions: float
	):
		"""
		Initialize the MultiActionExecutor.
		
		Args:
			browser_session: The browser session for DOM operations
			controller: The controller for executing actions
			action_model_class: The action model class to use
			page_extraction_llm: LLM for page extraction (optional)
			sensitive_data: Sensitive data dictionary (optional)
			available_file_paths: Available file paths (optional)
			generic_controller_context: Generic context for controller
			unexpected_elements_threshold: Threshold for detecting page changes
			wait_between_actions: Delay between actions in seconds
		"""
		self.browser_session = browser_session
		self.controller = controller
		self.action_model_class = action_model_class
		self.page_extraction_llm = page_extraction_llm
		self.sensitive_data = sensitive_data
		self.available_file_paths = available_file_paths
		self.generic_controller_context = generic_controller_context
		self.unexpected_elements_threshold = unexpected_elements_threshold
		self.wait_between_actions = wait_between_actions

	def update_dynamic_settings(
		self,
		sensitive_data: Optional[Dict[str, str]] = None,
		unexpected_elements_threshold: Optional[int] = None
	) -> None:
		"""
		Update dynamic settings that can change during execution.
		
		Args:
			sensitive_data: Updated sensitive data dictionary
			unexpected_elements_threshold: Updated threshold for detecting page changes
		"""
		if sensitive_data is not None:
			self.sensitive_data = sensitive_data
		if unexpected_elements_threshold is not None:
			self.unexpected_elements_threshold = unexpected_elements_threshold

	@time_execution_async('--multi-action-executor')
	async def execute(
		self,
		actions: List[AgentActionModel],
		check_for_new_elements: bool = True,
	) -> List[AgentActionResult]:
		"""
		Execute a sequence of actions with element relocation and stability checks.
		
		Args:
			actions: List of actions to execute
			check_for_new_elements: Whether to check for new elements after actions
			
		Returns:
			List of action results
		"""
		if not actions:
			return []

		# Get initial state for element relocation
		initial_state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
		initial_cached_selector_map = initial_state.selector_map
		
		# Create hash map for element relocation
		initial_cached_path_hashes = {}
		for index, element in initial_cached_selector_map.items():
			initial_cached_path_hashes[index] = HistoryTreeProcessor.convert_dom_element_to_history_element(element)

		results = []
		
		for i, action in enumerate(actions):
			logger.info(f'🎬 Executing action {i+1}/{len(actions)}: {action}')
			
			try:
				# Wait between actions if specified
				if i > 0 and self.wait_between_actions > 0:
					await asyncio.sleep(self.wait_between_actions)

				# Re-perceive the page state before each action (except the first one)
				if i > 0:
					await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)

				# Attempt element relocation if this action involves an element index
				await self._relocate_elements_if_needed(action, initial_cached_selector_map, initial_cached_path_hashes)

				# Check if we have sensitive data for this action
				has_sensitive_data = self._action_contains_sensitive_data(action)

				# Execute the action
				result = await self.controller.act(
					action=action,
					browser_session=self.browser_session,
					page_extraction_llm=self.page_extraction_llm,
					sensitive_data=self.sensitive_data,
					available_file_paths=self.available_file_paths,
					context=self.generic_controller_context,
					has_sensitive_data=has_sensitive_data,
				)

				results.append(result)
				
				# Log the result
				if result.error:
					logger.error(f'❌ Action {i+1} failed: {result.error}')
				else:
					logger.info(f'✅ Action {i+1} completed: {result.extracted_content}')

				# If this action completed the task, stop executing further actions
				if result.is_done:
					logger.info(f'🏁 Task completed after action {i+1}')
					break

			except Exception as e:
				error_message = f'Action {i+1} failed with exception: {str(e)}'
				logger.error(error_message)
				results.append(AgentActionResult(error=error_message))
				# Continue with next action instead of breaking

		# Global stability check after all actions
		if check_for_new_elements and results:
			await self._check_page_stability()

		return results

	async def _relocate_elements_if_needed(
		self,
		action: AgentActionModel,
		initial_cached_selector_map: Dict[int, DOMElementNode],
		initial_cached_path_hashes: Dict[int, Any]
	) -> None:
		"""
		Relocate elements if the page structure has changed since initial perception.
		
		Args:
			action: The action to potentially modify
			initial_cached_selector_map: Initial selector map
			initial_cached_path_hashes: Initial element hashes for relocation
		"""
		# Extract element index from action if it exists
		element_index = self._extract_element_index_from_action(action)
		if element_index is None:
			return

		# Check if element still exists at the same index
		current_state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
		current_selector_map = current_state.selector_map

		if element_index in current_selector_map:
			current_element = current_selector_map[element_index]
			if element_index in initial_cached_path_hashes:
				initial_history_element = initial_cached_path_hashes[element_index]
				
				# Check if element matches using full hash comparison
				if HistoryTreeProcessor.compare_history_element_and_dom_element(initial_history_element, current_element):
					# Element is the same, no relocation needed
					return

		# Element has moved or changed, attempt relocation
		if element_index in initial_cached_selector_map:
			initial_element = initial_cached_selector_map[element_index]
			await self._attempt_element_relocation(action, initial_element, current_state.element_tree)

	def _extract_element_index_from_action(self, action: AgentActionModel) -> Optional[int]:
		"""
		Extract element index from action if it contains one.
		
		Args:
			action: The action to examine
			
		Returns:
			Element index if found, None otherwise
		"""
		# Convert action to dict to examine its parameters
		action_dict = action.model_dump(exclude_unset=True)
		
		for action_name, params in action_dict.items():
			if params is not None and isinstance(params, dict):
				if 'index' in params:
					return params['index']
		
		return None

	async def _attempt_element_relocation(
		self,
		action: AgentActionModel,
		initial_element: DOMElementNode,
		current_tree: DOMElementNode
	) -> None:
		"""
		Attempt to relocate an element that has moved in the DOM.
		
		Args:
			action: The action to update with new element index
			initial_element: The original element we're looking for
			current_tree: Current DOM tree to search in
		"""
		# Convert initial element to history element for comparison
		initial_history_element = HistoryTreeProcessor.convert_dom_element_to_history_element(initial_element)
		
		# Try to find the element in the current tree using full hash matching
		relocated_element = HistoryTreeProcessor.find_history_element_in_tree(initial_history_element, current_tree)
		
		if relocated_element and relocated_element.highlight_index is not None:
			# Update action with new element index
			self._update_action_element_index(action, relocated_element.highlight_index)
			
			initial_label = get_effective_element_label(initial_element)
			new_label = get_effective_element_label(relocated_element)
			
			logger.info(
				f'🔄 Element relocated: "{initial_label}" -> index {relocated_element.highlight_index} ("{new_label}")'
			)
		else:
			# Try semantic matching as fallback
			await self._attempt_semantic_relocation(action, initial_element, current_tree)

	async def _attempt_semantic_relocation(
		self,
		action: AgentActionModel,
		initial_element: DOMElementNode,
		current_tree: DOMElementNode
	) -> None:
		"""
		Attempt semantic relocation when exact matching fails.
		
		Args:
			action: The action to update
			initial_element: The original element
			current_tree: Current DOM tree to search in
		"""
		initial_label = get_effective_element_label(initial_element)
		
		if not initial_label:
			logger.warning('🔍 Cannot perform semantic relocation: no label for original element')
			return

		# Find elements with similar labels
		best_match = None
		best_score = 0
		
		def search_tree(node: DOMElementNode) -> None:
			nonlocal best_match, best_score
			
			if node.highlight_index is not None:
				current_label = get_effective_element_label(node)
				
				# Simple similarity scoring (could be enhanced)
				if current_label and initial_label.lower() in current_label.lower():
					score = len(initial_label) / len(current_label) if current_label else 0
					if score > best_score:
						best_score = score
						best_match = node
			
			for child in node.children:
				if isinstance(child, DOMElementNode):
					search_tree(child)

		search_tree(current_tree)
		
		if best_match and best_match.highlight_index is not None and best_score > 0.5:
			self._update_action_element_index(action, best_match.highlight_index)
			new_label = get_effective_element_label(best_match)
			logger.info(
				f'🔍 Semantic relocation: "{initial_label}" -> index {best_match.highlight_index} ("{new_label}") '
				f'(similarity: {best_score:.2f})'
			)
		else:
			logger.warning(f'❌ Failed to relocate element: "{initial_label}"')

	def _update_action_element_index(self, action: AgentActionModel, new_index: int) -> None:
		"""
		Update the element index in an action.
		
		Args:
			action: The action to update
			new_index: The new element index
		"""
		# Convert action to dict, update index, and reconstruct
		action_dict = action.model_dump(exclude_unset=True)
		
		for action_name, params in action_dict.items():
			if params is not None and isinstance(params, dict) and 'index' in params:
				params['index'] = new_index
				# Update the action object in place
				setattr(action, action_name, self.action_model_class.model_validate(params))
				break

	def _action_contains_sensitive_data(self, action: AgentActionModel) -> bool:
		"""
		Check if an action contains sensitive data.
		
		Args:
			action: The action to check
			
		Returns:
			True if action contains sensitive data, False otherwise
		"""
		if not self.sensitive_data:
			return False

		# Convert action to dict to examine its parameters
		action_dict = action.model_dump(exclude_unset=True)
		
		for action_name, params in action_dict.items():
			if params is not None and isinstance(params, dict):
				if 'text' in params:
					text_value = params['text']
					if text_value and any(sensitive_value in text_value for sensitive_value in self.sensitive_data.values()):
						return True
		
		return False

	async def _check_page_stability(self) -> None:
		"""
		Check if too many new elements appeared on the page, indicating instability.
		"""
		try:
			current_state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)
			
			# Count new elements
			new_element_count = 0
			
			def count_new_elements(node: DOMElementNode) -> None:
				nonlocal new_element_count
				if node.highlight_index is not None and getattr(node, 'is_new', False):
					new_element_count += 1
				for child in node.children:
					if isinstance(child, DOMElementNode):
						count_new_elements(child)
			
			count_new_elements(current_state.element_tree)
			
			if new_element_count > self.unexpected_elements_threshold:
				logger.warning(
					f'⚠️ Page stability check: {new_element_count} new elements detected '
					f'(threshold: {self.unexpected_elements_threshold}). Page may have changed significantly.'
				)
			else:
				logger.debug(f'✅ Page stability check: {new_element_count} new elements (within threshold)')
				
		except Exception as e:
			logger.debug(f'Page stability check failed: {e}')