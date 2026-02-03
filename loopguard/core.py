"""Core loop detection logic."""

import asyncio
import hashlib
import inspect
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

T = TypeVar("T")


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
    """
    Create a hash signature from function arguments.

    Uses SHA-256 truncated to 16 chars for fast, collision-resistant hashing.
    Falls back to repr() for unhashable types.
    """
    try:
        sig_data = str((args, sorted(kwargs.items())))
    except TypeError:
        sig_data = repr((args, kwargs))
    return hashlib.sha256(sig_data.encode()).hexdigest()[:16]


def loopguard(
    max_repeats: int = 3,
    window: int = 60,
    on_loop: Optional[Callable[[Callable, tuple, dict], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to detect and prevent infinite loops in AI agents.

    Thread-safe. Memory-safe (auto-cleans old signatures).

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
    lock = threading.Lock()
    call_count = 0

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal call_count
            sig = _make_signature(args, kwargs)
            now = time.time()
            should_cleanup = False
            loop_detected = False

            with lock:
                # Clean old entries outside window
                call_history[sig] = [t for t in call_history[sig] if now - t < window]

                # Check for loop
                if len(call_history[sig]) >= max_repeats:
                    loop_detected = True
                else:
                    call_history[sig].append(now)
                    call_count += 1
                    if call_count % 100 == 0:
                        should_cleanup = True

            # Handle loop outside lock to avoid blocking
            if loop_detected:
                if on_loop is not None:
                    return cast(T, on_loop(func, args, kwargs))
                raise LoopDetectedError(func.__name__, max_repeats, window)

            # Periodic cleanup of old signatures
            if should_cleanup:
                _cleanup_old_signatures(call_history, window, lock)

            return func(*args, **kwargs)

        def reset() -> None:
            """Clear all call history."""
            with lock:
                call_history.clear()

        def get_count(sig_args: tuple = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments."""
            if sig_kwargs is None:
                sig_kwargs = {}
            sig = _make_signature(sig_args, sig_kwargs)
            now = time.time()
            with lock:
                return len([t for t in call_history[sig] if now - t < window])

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _cleanup_old_signatures(
    history: dict[str, list[float]],
    window: int,
    lock: threading.Lock,
) -> None:
    """Remove signatures that haven't been seen within 2x the window."""
    now = time.time()
    cutoff = now - (window * 2)

    with lock:
        to_remove = [
            key for key, times in history.items()
            if not times or max(times) < cutoff
        ]
        for key in to_remove:
            del history[key]


def async_loopguard(
    max_repeats: int = 3,
    window: int = 60,
    on_loop: Optional[Callable[[Callable, tuple, dict], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async version of loopguard decorator.

    Coroutine-safe. Memory-safe (auto-cleans old signatures).

    Args:
        max_repeats: Maximum allowed calls with same arguments within window.
        window: Time window in seconds.
        on_loop: Optional callback(func, args, kwargs) called when loop detected.
                 Can be sync or async function.

    Example:
        @async_loopguard(max_repeats=3, window=60)
        async def agent_action(query):
            return await llm.complete(query)
    """
    call_history: dict[str, list[float]] = defaultdict(list)
    lock = asyncio.Lock()
    sync_lock = threading.Lock()  # For sync operations like reset
    call_count = 0

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal call_count
            sig = _make_signature(args, kwargs)
            now = time.time()
            should_cleanup = False
            loop_detected = False

            async with lock:
                # Clean old entries outside window
                call_history[sig] = [t for t in call_history[sig] if now - t < window]

                # Check for loop
                if len(call_history[sig]) >= max_repeats:
                    loop_detected = True
                else:
                    call_history[sig].append(now)
                    call_count += 1
                    if call_count % 100 == 0:
                        should_cleanup = True

            # Handle loop outside lock
            if loop_detected:
                if on_loop is not None:
                    result = on_loop(func, args, kwargs)
                    if inspect.iscoroutine(result):
                        return cast(T, await result)
                    return cast(T, result)
                raise LoopDetectedError(func.__name__, max_repeats, window)

            # Periodic cleanup
            if should_cleanup:
                async with lock:
                    cleanup_now = time.time()
                    cutoff = cleanup_now - (window * 2)
                    to_remove = [
                        key for key, times in call_history.items()
                        if not times or max(times) < cutoff
                    ]
                    for key in to_remove:
                        del call_history[key]

            return await func(*args, **kwargs)

        def reset() -> None:
            """Clear all call history (sync, safe to call from any context)."""
            with sync_lock:
                call_history.clear()

        def get_count(sig_args: tuple = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments (sync)."""
            if sig_kwargs is None:
                sig_kwargs = {}
            sig = _make_signature(sig_args, sig_kwargs)
            now = time.time()
            with sync_lock:
                return len([t for t in call_history[sig] if now - t < window])

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        return cast(Callable[..., T], wrapper)

    return decorator
