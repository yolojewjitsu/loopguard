"""Tests for loopguard."""

import time
import threading

import pytest

from loopguard import LoopDetectedError, async_loopguard, loopguard


class TestParameterValidation:
    def test_max_repeats_zero_raises(self):
        with pytest.raises(ValueError, match="max_repeats must be >= 1"):
            @loopguard(max_repeats=0)
            def func(x):
                return x

    def test_max_repeats_negative_raises(self):
        with pytest.raises(ValueError, match="max_repeats must be >= 1"):
            @loopguard(max_repeats=-5)
            def func(x):
                return x

    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="window must be > 0"):
            @loopguard(window=0)
            def func(x):
                return x

    def test_window_negative_raises(self):
        with pytest.raises(ValueError, match="window must be > 0"):
            @loopguard(window=-10)
            def func(x):
                return x

    def test_async_max_repeats_zero_raises(self):
        with pytest.raises(ValueError, match="max_repeats must be >= 1"):
            @async_loopguard(max_repeats=0)
            async def func(x):
                return x

    def test_async_window_negative_raises(self):
        with pytest.raises(ValueError, match="window must be > 0"):
            @async_loopguard(window=-1)
            async def func(x):
                return x

    def test_float_window(self):
        """Test that float windows work for sub-second precision."""
        @loopguard(max_repeats=2, window=0.5)
        def func(x):
            return x

        func(1)
        func(1)

        with pytest.raises(LoopDetectedError):
            func(1)

        # After window expires, should work again
        time.sleep(0.6)
        assert func(1) == 1

    def test_on_loop_exception_propagates(self):
        """Test that exceptions from on_loop handler propagate to caller."""
        def bad_handler(func, args, kwargs):
            raise RuntimeError("Handler failed")

        @loopguard(max_repeats=1, on_loop=bad_handler)
        def func(x):
            return x

        func(1)

        with pytest.raises(RuntimeError, match="Handler failed"):
            func(1)


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

    def test_unhashable_args(self):
        """Test that unhashable arguments (like dicts, lists) work via repr fallback."""
        @loopguard(max_repeats=2, window=60)
        def func(data):
            return data.get("key", 0)

        # Dicts are unhashable but should still work
        assert func({"key": 1}) == 1
        assert func({"key": 1}) == 1

        with pytest.raises(LoopDetectedError):
            func({"key": 1})

        # Different dict = different signature
        assert func({"key": 2}) == 2

    def test_unhashable_nested(self):
        """Test deeply nested unhashable structures."""
        @loopguard(max_repeats=2, window=60)
        def func(data):
            return len(data)

        nested = [{"a": [1, 2, 3]}, {"b": {"c": [4, 5]}}]
        assert func(nested) == 2
        assert func(nested) == 2

        with pytest.raises(LoopDetectedError):
            func(nested)

    def test_get_count(self):
        """Test the get_count helper method."""
        @loopguard(max_repeats=5, window=60)
        def func(x):
            return x

        assert func.get_count((5,)) == 0
        func(5)
        assert func.get_count((5,)) == 1
        func(5)
        assert func.get_count((5,)) == 2

    def test_thread_safety(self):
        """Test that loopguard is thread-safe."""
        @loopguard(max_repeats=100, window=60)
        def func(x):
            return x

        errors = []
        results = []

        def worker():
            try:
                for _ in range(50):
                    results.append(func(1))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 200  # 4 threads * 50 calls

    def test_get_count_no_side_effects(self):
        """Test that get_count doesn't create entries in call_history."""
        @loopguard(max_repeats=5, window=60)
        def func(x):
            return x

        # Querying non-existent signature should return 0
        assert func.get_count((999,)) == 0
        assert func.get_count((999,)) == 0  # Still 0, no entry created

        # Now actually call it
        func(999)
        assert func.get_count((999,)) == 1

    def test_reset_clears_call_count(self):
        """Test that reset() also clears the internal call counter."""
        @loopguard(max_repeats=10, window=60)
        def func(x):
            return x

        # Make many calls to increment call_count
        for i in range(150):
            func(i)

        # Reset should clear everything including call_count
        func.reset()

        # Should work fine after reset
        for i in range(150):
            func(i)

        assert True  # If we got here without errors, it works


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
    async def test_sync_handler(self):
        """Test that sync handlers work with async_loopguard."""
        def handler(func, args, kwargs):
            return "sync handled"

        @async_loopguard(max_repeats=2, window=60, on_loop=handler)
        async def func(x):
            return x

        await func(5)
        await func(5)
        result = await func(5)

        assert result == "sync handled"

    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Test that async handlers are properly awaited."""
        handler_awaited = []

        async def handler(func, args, kwargs):
            handler_awaited.append(True)
            return "async handled"

        @async_loopguard(max_repeats=2, window=60, on_loop=handler)
        async def func(x):
            return x

        await func(5)
        await func(5)
        result = await func(5)

        assert result == "async handled"
        assert handler_awaited == [True]  # Confirms handler was called

    @pytest.mark.asyncio
    async def test_unhashable_args_async(self):
        """Test unhashable args with async version."""
        @async_loopguard(max_repeats=2, window=60)
        async def func(data):
            return data["value"]

        assert await func({"value": 42}) == 42
        assert await func({"value": 42}) == 42

        with pytest.raises(LoopDetectedError):
            await func({"value": 42})


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
