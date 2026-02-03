"""Core loop detection logic."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, cast

__all__ = ["loopguard", "async_loopguard", "LoopDetectedError"]

T = TypeVar("T")


class LoopDetectedError(Exception):
    """Raised when a loop is detected."""

    def __init__(self, func_name: str, count: int, window: float):
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


def _filter_recent(timestamps: list[float], cutoff: float) -> list[float]:
    """Filter timestamps to only those after cutoff. Returns new list."""
    return [t for t in timestamps if t >= cutoff]


def _cleanup_old_signatures(
    history: dict[str, list[float]],
    window: float,
    lock: threading.Lock,
) -> None:
    """Remove signatures that haven't been seen within 2x the window."""
    now = time.time()
    cutoff = now - (window * 2)

    with lock:
        to_remove = [
            key for key, times in history.items()
            if len(times) == 0 or times[-1] < cutoff  # times is sorted, check last
        ]
        for key in to_remove:
            del history[key]


def loopguard(
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to detect and prevent infinite loops in AI agents.

    Thread-safe. Memory-safe (auto-cleans old signatures).

    Args:
        max_repeats: Maximum allowed calls with same arguments within window (must be >= 1).
        window: Time window in seconds (must be > 0). Can be float for sub-second precision.
        on_loop: Optional callback called when loop detected.
                 Signature: on_loop(func, args, kwargs) -> Any
                 If provided, its return value is used instead of raising.
                 If the callback raises an exception, it propagates to the caller.

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
    cleanup_counter = 0  # Only tracks cleanup scheduling, resets at 100

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal cleanup_counter
            sig = _make_signature(args, kwargs)
            now = time.time()
            cutoff = now - window
            should_cleanup = False
            loop_detected = False

            with lock:
                # Get existing history or empty
                timestamps = call_history.get(sig)

                if timestamps is None:
                    # New signature, just record it
                    call_history[sig] = [now]
                else:
                    # Filter to recent calls only
                    recent = _filter_recent(timestamps, cutoff)

                    # Check for loop
                    if len(recent) >= max_repeats:
                        loop_detected = True
                    else:
                        recent.append(now)
                        call_history[sig] = recent

                if not loop_detected:
                    cleanup_counter = (cleanup_counter + 1) % 100
                    if cleanup_counter == 0:
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
            nonlocal cleanup_counter
            with lock:
                call_history.clear()
                cleanup_counter = 0

        def get_count(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments."""
            if sig_kwargs is None:
                sig_kwargs = {}
            sig = _make_signature(sig_args, sig_kwargs)
            now = time.time()
            cutoff = now - window
            with lock:
                timestamps = call_history.get(sig)
                if timestamps is None:
                    return 0
                return len(_filter_recent(timestamps, cutoff))

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        return wrapper

    return decorator


def async_loopguard(
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async version of loopguard decorator.

    Coroutine-safe. Memory-safe (auto-cleans old signatures).

    Args:
        max_repeats: Maximum allowed calls with same arguments within window (must be >= 1).
        window: Time window in seconds (must be > 0). Can be float for sub-second precision.
        on_loop: Optional callback called when loop detected.
                 Signature: on_loop(func, args, kwargs) -> Any
                 Can be sync or async function.
                 If the callback raises an exception, it propagates to the caller.

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
    # Single lock for both sync and async access - simpler and avoids deadlock
    lock = threading.Lock()
    lock_init = threading.Lock()  # Protects async_lock initialization
    async_lock: Optional[asyncio.Lock] = None
    cleanup_counter = 0

    def _get_async_lock() -> asyncio.Lock:
        """Get or create the async lock (thread-safe lazy initialization)."""
        nonlocal async_lock
        if async_lock is None:
            with lock_init:
                if async_lock is None:  # Double-check pattern
                    async_lock = asyncio.Lock()
        return async_lock

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal cleanup_counter
            sig = _make_signature(args, kwargs)
            now = time.time()
            cutoff = now - window
            should_cleanup = False
            loop_detected = False

            # Use async lock for coroutine ordering, threading lock for data protection
            alock = _get_async_lock()
            async with alock:
                with lock:
                    timestamps = call_history.get(sig)

                    if timestamps is None:
                        call_history[sig] = [now]
                    else:
                        recent = _filter_recent(timestamps, cutoff)

                        if len(recent) >= max_repeats:
                            loop_detected = True
                        else:
                            recent.append(now)
                            call_history[sig] = recent

                    if not loop_detected:
                        cleanup_counter = (cleanup_counter + 1) % 100
                        if cleanup_counter == 0:
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
                _cleanup_old_signatures(call_history, window, lock)

            return await func(*args, **kwargs)

        def reset() -> None:
            """Clear all call history (sync, safe to call from any context)."""
            nonlocal cleanup_counter
            with lock:
                call_history.clear()
                cleanup_counter = 0

        def get_count(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments (sync)."""
            if sig_kwargs is None:
                sig_kwargs = {}
            sig = _make_signature(sig_args, sig_kwargs)
            now = time.time()
            cutoff = now - window
            with lock:
                timestamps = call_history.get(sig)
                if timestamps is None:
                    return 0
                return len(_filter_recent(timestamps, cutoff))

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        return cast(Callable[..., T], wrapper)

    return decorator
