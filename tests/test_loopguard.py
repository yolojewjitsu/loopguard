"""Tests for loopguard."""

import asyncio
import time

import pytest

from loopguard import LoopDetectedError, async_loopguard, loopguard


class TestLoopguard:
    def test_allows_calls_under_limit(self):
        @loopguard(max_repeats=3, window=60)
        def func(x):
            return x * 2

        assert func(5) == 10
        assert func(5) == 10
        assert func(5) == 10  # 3rd call is allowed

    def test_raises_on_loop(self):
        @loopguard(max_repeats=3, window=60)
        def func(x):
            return x * 2

        func(5)
        func(5)
        func(5)

        with pytest.raises(LoopDetectedError) as exc:
            func(5)  # 4th call raises

        assert exc.value.func_name == "func"
        assert exc.value.count == 3

    def test_different_args_dont_trigger(self):
        @loopguard(max_repeats=2, window=60)
        def func(x):
            return x

        # Different args = different signatures
        assert func(1) == 1
        assert func(2) == 2
        assert func(3) == 3
        assert func(1) == 1  # Still under limit for arg=1

    def test_window_expiry(self):
        @loopguard(max_repeats=2, window=1)
        def func(x):
            return x

        func(5)
        func(5)

        # Wait for window to expire
        time.sleep(1.1)

        # Should work again
        assert func(5) == 5

    def test_custom_handler(self):
        handler_called = []

        def handler(func, args, kwargs):
            handler_called.append((func.__name__, args))
            return "handled"

        @loopguard(max_repeats=2, window=60, on_loop=handler)
        def func(x):
            return x * 2

        func(5)
        func(5)
        result = func(5)  # Triggers handler

        assert result == "handled"
        assert handler_called == [("func", (5,))]

    def test_reset(self):
        @loopguard(max_repeats=2, window=60)
        def func(x):
            return x

        func(5)
        func(5)

        func.reset()

        # Should work again after reset
        assert func(5) == 5

    def test_kwargs_in_signature(self):
        @loopguard(max_repeats=2, window=60)
        def func(x, y=1):
            return x + y

        func(1, y=2)
        func(1, y=2)

        with pytest.raises(LoopDetectedError):
            func(1, y=2)

        # Different kwarg value = different signature
        assert func(1, y=3) == 4


class TestAsyncLoopguard:
    @pytest.mark.asyncio
    async def test_allows_calls_under_limit(self):
        @async_loopguard(max_repeats=3, window=60)
        async def func(x):
            return x * 2

        assert await func(5) == 10
        assert await func(5) == 10

    @pytest.mark.asyncio
    async def test_raises_on_loop(self):
        @async_loopguard(max_repeats=2, window=60)
        async def func(x):
            return x

        await func(5)
        await func(5)

        with pytest.raises(LoopDetectedError):
            await func(5)

    @pytest.mark.asyncio
    async def test_async_handler(self):
        async def handler(func, args, kwargs):
            return "async handled"

        @async_loopguard(max_repeats=2, window=60, on_loop=handler)
        async def func(x):
            return x

        await func(5)
        await func(5)
        result = await func(5)

        assert result == "async handled"


class TestLoopDetectedError:
    def test_error_message(self):
        err = LoopDetectedError("my_func", 5, 120)
        assert "my_func" in str(err)
        assert "5" in str(err)
        assert "120" in str(err)

    def test_error_attributes(self):
        err = LoopDetectedError("test", 3, 60)
        assert err.func_name == "test"
        assert err.count == 3
        assert err.window == 60
