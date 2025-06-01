from .profile import (
    CHROME_DEFAULT_ARGS,
    CHROME_DETERMINISTIC_RENDERING_ARGS,
    CHROME_DISABLE_SECURITY_ARGS,
    CHROME_DOCKER_ARGS,
    CHROME_HEADLESS_ARGS,
    BrowserProfile,
)
from .session import BrowserSession

BrowserConfig = BrowserProfile
BrowserContextConfig = BrowserProfile
Browser = BrowserSession
CHROME_ARGS = CHROME_DEFAULT_ARGS
# CHROME_DETERMINISTIC_RENDERING_ARGS is also expected by the web-ui
# CHROME_DISABLE_SECURITY_ARGS is also expected by the web-ui
# CHROME_DOCKER_ARGS is also expected by the web-ui
# CHROME_HEADLESS_ARGS is also expected by the web-ui

__all__ = [
    'BrowserProfile',
    'BrowserSession',
    'BrowserConfig',
    'BrowserContextConfig',
    'Browser',
    'CHROME_ARGS',
    'CHROME_DETERMINISTIC_RENDERING_ARGS',
    'CHROME_DISABLE_SECURITY_ARGS',
    'CHROME_DOCKER_ARGS',
    'CHROME_HEADLESS_ARGS',
]