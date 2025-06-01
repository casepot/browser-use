import os

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

# Check if running in Docker
IN_DOCKER = os.environ.get('IN_DOCKER', 'false').lower()[0] in 'ty1'

BrowserConfig = BrowserProfile
BrowserContextConfig = BrowserProfile
Browser = BrowserSession

__all__ = ['BrowserConfig', 'BrowserContextConfig', 'Browser', 'IN_DOCKER']
