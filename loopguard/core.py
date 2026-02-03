"""Core loop detection logic."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
import time
from collections import deque
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, cast, overload

__all__ = ["loopguard", "async_loopguard", "LoopDetectedError"]

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Use monotonic time for internal tracking to avoid issues with system clock changes.
_get_time = time.monotonic


class LoopDetectedError(Exception):
    """Raised when a loop is detected.

    Attributes:
        func_name: Name of the function that triggered the loop detection.
        count: The max_repeats threshold that was exceeded.
        window: The time window in seconds.
    """

    def __init__(self, func_name: str, count: int, window: Union[int, float]):
        self.func_name = func_name
        self.count = count
        self.window = window
        super().__init__(
            f"Loop detected: {func_name} called {count}+ times with same args in {window}s"
        )

    def __repr__(self) -> str:
        return f"LoopDetectedError({self.func_name!r}, count={self.count}, window={self.window})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LoopDetectedError):
            return NotImplemented
        return (
            self.func_name == other.func_name
            and self.count == other.count
            and self.window == other.window
        )

    def __hash__(self) -> int:
        return hash((self.func_name, self.count, self.window))


def _make_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """
    Create a hash signature from function arguments.

    Uses SHA-256 truncated to 16 hex chars (64 bits) for fast hashing.
    Collision probability is negligible for typical usage (< millions of unique signatures).

    Falls back to repr() for unhashable types. Note: repr() may include memory
    addresses for some objects, causing logically equal objects to have different
    signatures. For reliable behavior, ensure arguments are value-comparable.
    """
    try:
        sig_data = str((args, sorted(kwargs.items())))
    except TypeError:
        sig_data = repr((args, kwargs))
    return hashlib.sha256(sig_data.encode()).hexdigest()[:16]


def _filter_recent_deque(timestamps: deque[float], cutoff: float) -> None:
    """Filter timestamps deque in-place, keeping only those >= cutoff. O(k) where k = removed."""
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()


def _cleanup_old_signatures(
    history: dict[str, deque[float]],
    window: Union[int, float],
    lock: threading.Lock,
) -> None:
    """Remove signatures that haven't been seen within 2x the window."""
    now = _get_time()
    cutoff = now - (window * 2)

    with lock:
        to_remove = [
            key for key, times in history.items()
            if len(times) == 0 or times[-1] < cutoff
        ]
        for key in to_remove:
            del history[key]


def _validate_params(max_repeats: int, window: Union[int, float]) -> None:
    """Validate decorator parameters."""
    if max_repeats < 1:
        raise ValueError(f"max_repeats must be >= 1, got {max_repeats}")
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")


def _normalize_kwargs(sig_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Normalize kwargs argument, returning empty dict if None."""
    return {} if sig_kwargs is None else sig_kwargs


# Overloads for loopguard to support both @loopguard and @loopguard()
@overload
def loopguard(func: F) -> F: ...
@overload
def loopguard(
    func: None = None,
    *,
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
@overload
def loopguard(
    *,
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...


def loopguard(
    func: Optional[F] = None,
    *,
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Union[F, Callable[[Callable[..., T]], Callable[..., T]]]:
    """
    Decorator to detect and prevent infinite loops in AI agents.

    Thread-safe. Memory-safe (auto-cleans old signatures).

    Can be used with or without parentheses:
        @loopguard
        def func(): ...

        @loopguard()
        def func(): ...

        @loopguard(max_repeats=5)
        def func(): ...

    Args:
        max_repeats: Maximum allowed calls with same arguments within window (must be >= 1).
        window: Time window in seconds (must be > 0). Can be float for sub-second precision.
                Uses monotonic time internally, so immune to system clock changes.
        on_loop: Optional callback called when loop detected.
                 Signature: on_loop(func, args, kwargs) -> Any
                 The func passed is the original unwrapped function.
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

        # Sub-second window
        @loopguard(max_repeats=5, window=0.5)
        def rate_limited_call(x):
            return api.call(x)
    """
    _validate_params(max_repeats, window)

    call_history: dict[str, deque[float]] = {}
    lock = threading.Lock()
    cleanup_counter = 0

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal cleanup_counter
            sig = _make_signature(args, kwargs)
            now = _get_time()
            cutoff = now - window
            should_cleanup = False
            loop_detected = False

            with lock:
                timestamps = call_history.get(sig)

                if timestamps is None:
                    call_history[sig] = deque([now])
                else:
                    _filter_recent_deque(timestamps, cutoff)

                    if len(timestamps) >= max_repeats:
                        loop_detected = True
                    else:
                        timestamps.append(now)

                if not loop_detected:
                    cleanup_counter = (cleanup_counter + 1) % 100
                    if cleanup_counter == 0:
                        should_cleanup = True

            if loop_detected:
                if on_loop is not None:
                    return cast(T, on_loop(fn, args, kwargs))
                raise LoopDetectedError(fn.__name__, max_repeats, window)

            if should_cleanup:
                _cleanup_old_signatures(call_history, window, lock)

            return fn(*args, **kwargs)

        def reset() -> None:
            """Clear all call history."""
            nonlocal cleanup_counter
            with lock:
                call_history.clear()
                cleanup_counter = 0

        def get_count(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments."""
            sig = _make_signature(sig_args, _normalize_kwargs(sig_kwargs))
            now = _get_time()
            cutoff = now - window
            with lock:
                timestamps = call_history.get(sig)
                if timestamps is None:
                    return 0
                return sum(1 for t in timestamps if t >= cutoff)

        def would_trigger(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> bool:
            """Check if calling with these arguments would trigger loop detection."""
            return get_count(sig_args, _normalize_kwargs(sig_kwargs)) >= max_repeats

        def get_signatures() -> list[str]:
            """Get list of all tracked signatures (for debugging)."""
            with lock:
                return list(call_history.keys())

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        wrapper.would_trigger = would_trigger  # type: ignore[attr-defined]
        wrapper.get_signatures = get_signatures  # type: ignore[attr-defined]
        return wrapper

    # Support @loopguard without parentheses
    if func is not None:
        return decorator(func)
    return decorator


# Overloads for async_loopguard
@overload
def async_loopguard(func: F) -> F: ...
@overload
def async_loopguard(
    func: None = None,
    *,
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
@overload
def async_loopguard(
    *,
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...


def async_loopguard(
    func: Optional[F] = None,
    *,
    max_repeats: int = 3,
    window: Union[int, float] = 60,
    on_loop: Optional[Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]] = None,
) -> Union[F, Callable[[Callable[..., T]], Callable[..., T]]]:
    """
    Async version of loopguard decorator.

    Coroutine-safe. Memory-safe (auto-cleans old signatures).

    Can be used with or without parentheses:
        @async_loopguard
        async def func(): ...

        @async_loopguard()
        async def func(): ...

    Args:
        max_repeats: Maximum allowed calls with same arguments within window (must be >= 1).
        window: Time window in seconds (must be > 0). Can be float for sub-second precision.
                Uses monotonic time internally, so immune to system clock changes.
        on_loop: Optional callback called when loop detected.
                 Signature: on_loop(func, args, kwargs) -> Any
                 The func passed is the original unwrapped function.
                 Can be sync or async function.
                 If the callback raises an exception, it propagates to the caller.

    Raises:
        ValueError: If max_repeats < 1 or window <= 0.

    Example:
        @async_loopguard(max_repeats=3, window=60)
        async def agent_action(query):
            return await llm.complete(query)

        # Sub-second window
        @async_loopguard(max_repeats=10, window=1.0)
        async def rate_limited_api(x):
            return await api.call(x)
    """
    _validate_params(max_repeats, window)

    call_history: dict[str, deque[float]] = {}
    lock = threading.Lock()
    lock_init = threading.Lock()  # Protects async lock initialization
    cleanup_counter = 0
    async_lock: Optional[asyncio.Lock] = None

    def _get_async_lock() -> asyncio.Lock:
        """Get or create async lock (thread-safe lazy initialization)."""
        nonlocal async_lock
        if async_lock is None:
            with lock_init:
                if async_lock is None:
                    async_lock = asyncio.Lock()
        return async_lock

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal cleanup_counter
            sig = _make_signature(args, kwargs)
            now = _get_time()
            cutoff = now - window
            should_cleanup = False
            loop_detected = False

            alock = _get_async_lock()
            async with alock:
                with lock:
                    timestamps = call_history.get(sig)

                    if timestamps is None:
                        call_history[sig] = deque([now])
                    else:
                        _filter_recent_deque(timestamps, cutoff)

                        if len(timestamps) >= max_repeats:
                            loop_detected = True
                        else:
                            timestamps.append(now)

                    if not loop_detected:
                        cleanup_counter = (cleanup_counter + 1) % 100
                        if cleanup_counter == 0:
                            should_cleanup = True

            if loop_detected:
                if on_loop is not None:
                    result = on_loop(fn, args, kwargs)
                    if inspect.iscoroutine(result):
                        return cast(T, await result)
                    return cast(T, result)
                raise LoopDetectedError(fn.__name__, max_repeats, window)

            if should_cleanup:
                _cleanup_old_signatures(call_history, window, lock)

            return await fn(*args, **kwargs)

        def reset() -> None:
            """Clear all call history (sync, safe to call from any context)."""
            nonlocal cleanup_counter
            with lock:
                call_history.clear()
                cleanup_counter = 0

        def get_count(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> int:
            """Get current call count for given arguments (sync)."""
            sig = _make_signature(sig_args, _normalize_kwargs(sig_kwargs))
            now = _get_time()
            cutoff = now - window
            with lock:
                timestamps = call_history.get(sig)
                if timestamps is None:
                    return 0
                return sum(1 for t in timestamps if t >= cutoff)

        def would_trigger(sig_args: tuple[Any, ...] = (), sig_kwargs: Optional[dict[str, Any]] = None) -> bool:
            """Check if calling with these arguments would trigger loop detection."""
            return get_count(sig_args, _normalize_kwargs(sig_kwargs)) >= max_repeats

        def get_signatures() -> list[str]:
            """Get list of all tracked signatures (for debugging)."""
            with lock:
                return list(call_history.keys())

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.get_count = get_count  # type: ignore[attr-defined]
        wrapper.would_trigger = would_trigger  # type: ignore[attr-defined]
        wrapper.get_signatures = get_signatures  # type: ignore[attr-defined]
        return cast(Callable[..., T], wrapper)

    # Support @async_loopguard without parentheses
    if func is not None:
        return decorator(func)
    return decorator
