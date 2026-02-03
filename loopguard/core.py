"""Core loop detection logic."""

from collections import defaultdict
from functools import wraps
import hashlib
import time
from typing import Any, Callable, Optional


class LoopDetectedError(Exception):
    """Raised when a loop is detected."""

    def __init__(self, func_name: str, count: int, window: int):
        self.func_name = func_name
        self.count = count
        self.window = window
        super().__init__(
            f"Loop detected: {func_name} called {count}+ times with same args in {window}s"
        )


def _make_signature(args: tuple, kwargs: dict) -> str:
    """Create a hash signature from function arguments."""
    # Handle unhashable types by converting to string
    try:
        sig_data = str((args, sorted(kwargs.items())))
    except TypeError:
        sig_data = repr((args, kwargs))
    return hashlib.md5(sig_data.encode()).hexdigest()


def loopguard(
    max_repeats: int = 3,
    window: int = 60,
    on_loop: Optional[Callable[..., Any]] = None,
):
    """
    Decorator to detect and prevent infinite loops in AI agents.

    Args:
        max_repeats: Maximum allowed calls with same arguments within window.
        window: Time window in seconds.
        on_loop: Optional callback(func, args, kwargs) called when loop detected.
                 If provided, its return value is used instead of raising.

    Example:
        @loopguard(max_repeats=3, window=60)
        def agent_action(query):
            return llm.complete(query)

        # With custom handler
        @loopguard(max_repeats=3, on_loop=lambda f, a, k: "Loop stopped")
        def agent_action(query):
            return llm.complete(query)
    """
    call_history: dict[str, list[float]] = defaultdict(list)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = _make_signature(args, kwargs)
            now = time.time()

            # Clean old entries outside window
            call_history[sig] = [t for t in call_history[sig] if now - t < window]

            # Check for loop
            if len(call_history[sig]) >= max_repeats:
                if on_loop is not None:
                    return on_loop(func, args, kwargs)
                raise LoopDetectedError(func.__name__, max_repeats, window)

            call_history[sig].append(now)
            return func(*args, **kwargs)

        # Expose reset for testing
        wrapper.reset = lambda: call_history.clear()  # type: ignore
        return wrapper

    return decorator


def async_loopguard(
    max_repeats: int = 3,
    window: int = 60,
    on_loop: Optional[Callable[..., Any]] = None,
):
    """
    Async version of loopguard decorator.

    Args:
        max_repeats: Maximum allowed calls with same arguments within window.
        window: Time window in seconds.
        on_loop: Optional async callback(func, args, kwargs) called when loop detected.

    Example:
        @async_loopguard(max_repeats=3, window=60)
        async def agent_action(query):
            return await llm.complete(query)
    """
    call_history: dict[str, list[float]] = defaultdict(list)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            sig = _make_signature(args, kwargs)
            now = time.time()

            # Clean old entries outside window
            call_history[sig] = [t for t in call_history[sig] if now - t < window]

            # Check for loop
            if len(call_history[sig]) >= max_repeats:
                if on_loop is not None:
                    if hasattr(on_loop, "__await__") or hasattr(
                        on_loop, "__call__"
                    ):
                        result = on_loop(func, args, kwargs)
                        if hasattr(result, "__await__"):
                            return await result
                        return result
                raise LoopDetectedError(func.__name__, max_repeats, window)

            call_history[sig].append(now)
            return await func(*args, **kwargs)

        wrapper.reset = lambda: call_history.clear()  # type: ignore
        return wrapper

    return decorator
