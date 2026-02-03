"""LoopGuard - Protect AI agents from infinite loops."""

from .core import loopguard, async_loopguard, LoopDetectedError

__version__ = "0.2.0"
__all__ = ["loopguard", "async_loopguard", "LoopDetectedError", "__version__"]
