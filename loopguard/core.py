"""Core loop detection logic."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

__all__ = ["loopguard", "async_loopguard", "LoopDetectedError"]

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


def _make_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
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


def loopguard(
    max_repeats: int = 3,
    window: int = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to detect and prevent infinite loops in AI agents.

    Thread-safe. Memory-safe (auto-cleans old signatures).

    Args:
        max_repeats: Maximum allowed calls with same arguments within window (must be >= 1).
        window: Time window in seconds (must be > 0).
        on_loop: Optional callback called when loop detected.
                 Signature: on_loop(func, args, kwargs) -> Any
                 If provided, its return value is used instead of raising.

    Raises:
        ValueError: If max_repeats < 1 or window <= 0.

    Example:
        @loopguard(max_repeats=3, window=60)
        def agent_action(query):
            return llm.complete(query)

        # With custom handler
        @loopguard(max_repeats=3, on_loop=lambda f, a, k: "Loop stopped")
        def agent_action(query):
            return llm.complete(query)
    """
    if max_repeats < 1:
        raise ValueError(f"max_repeats must be >= 1, got {max_repeats}")
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    call_history: dict[str, list[float]] = {}
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
                # Get or create history for this signature
                if sig not in call_history:
                    call_history[sig] = []

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
            nonlocal call_count
            with lock:
                call_history.clear()
                call_count = 0

        def get_count(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments."""
            if sig_kwargs is None:
                sig_kwargs = {}
            sig = _make_signature(sig_args, sig_kwargs)
            now = time.time()
            with lock:
                if sig not in call_history:
                    return 0
                return len([t for t in call_history[sig] if now - t < window])

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        return wrapper

    return decorator


def async_loopguard(
    max_repeats: int = 3,
    window: int = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async version of loopguard decorator.

    Coroutine-safe. Memory-safe (auto-cleans old signatures).

    Args:
        max_repeats: Maximum allowed calls with same arguments within window (must be >= 1).
        window: Time window in seconds (must be > 0).
        on_loop: Optional callback called when loop detected.
                 Signature: on_loop(func, args, kwargs) -> Any
                 Can be sync or async function.

    Raises:
        ValueError: If max_repeats < 1 or window <= 0.

    Example:
        @async_loopguard(max_repeats=3, window=60)
        async def agent_action(query):
            return await llm.complete(query)
    """
    if max_repeats < 1:
        raise ValueError(f"max_repeats must be >= 1, got {max_repeats}")
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    call_history: dict[str, list[float]] = {}
    lock: Optional[asyncio.Lock] = None  # Lazy init to avoid event loop issues
    rlock = threading.RLock()  # For all sync access to call_history
    call_count = 0

    def _get_async_lock() -> asyncio.Lock:
        """Get or create the async lock (lazy initialization)."""
        nonlocal lock
        if lock is None:
            lock = asyncio.Lock()
        return lock

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal call_count
            sig = _make_signature(args, kwargs)
            now = time.time()
            should_cleanup = False
            loop_detected = False

            async_lock = _get_async_lock()
            async with async_lock:
                with rlock:  # Protect dict access from sync callers
                    # Get or create history for this signature
                    if sig not in call_history:
                        call_history[sig] = []

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
                async with async_lock:
                    with rlock:
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
            nonlocal call_count
            with rlock:
                call_history.clear()
                call_count = 0

        def get_count(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments (sync)."""
            if sig_kwargs is None:
                sig_kwargs = {}
            sig = _make_signature(sig_args, sig_kwargs)
            now = time.time()
            with rlock:
                if sig not in call_history:
                    return 0
                return len([t for t in call_history[sig] if now - t < window])

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        return cast(Callable[..., T], wrapper)

    return decorator
