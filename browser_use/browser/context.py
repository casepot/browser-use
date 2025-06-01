from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

Browser = BrowserSession
BrowserConfig = BrowserProfile
BrowserContext = BrowserSession
BrowserContextConfig = BrowserProfile
BrowserContextState = BrowserSession  # Alias for compatibility

__all__ = ['Browser', 'BrowserConfig', 'BrowserContext', 'BrowserContextConfig', 'BrowserContextState']
