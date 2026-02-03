"""Tests for loopguard."""

import time
import threading

import pytest

from loopguard import LoopDetectedError, async_loopguard, loopguard


class TestDecoratorSyntax:
    def test_loopguard_without_parens(self):
        """Test @loopguard without parentheses."""
        @loopguard
        def func(x):
            return x

        func(1)
        func(1)
        func(1)  # Default max_repeats=3

        with pytest.raises(LoopDetectedError):
            func(1)

    def test_loopguard_empty_parens(self):
        """Test @loopguard() with empty parentheses."""
        @loopguard()
        def func(x):
            return x

        func(1)
        func(1)
        func(1)

        with pytest.raises(LoopDetectedError):
            func(1)

    @pytest.mark.asyncio
    async def test_async_loopguard_without_parens(self):
        """Test @async_loopguard without parentheses."""
        @async_loopguard
        async def func(x):
            return x

        await func(1)
        await func(1)
        await func(1)

        with pytest.raises(LoopDetectedError):
            await func(1)


class TestGetSignatures:
    def test_get_signatures_empty(self):
        @loopguard(max_repeats=5, window=60)
        def func(x):
            return x

        assert func.get_signatures() == []

    def test_get_signatures_tracks_calls(self):
        @loopguard(max_repeats=5, window=60)
        def func(x):
            return x

        func(1)
        func(2)
        func(3)

        sigs = func.get_signatures()
        assert len(sigs) == 3

    def test_get_signatures_same_args_one_sig(self):
        @loopguard(max_repeats=5, window=60)
        def func(x):
            return x

        func(1)
        func(1)
        func(1)

        sigs = func.get_signatures()
        assert len(sigs) == 1


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

    def test_async_handler_in_sync_loopguard_raises(self):
        """Test that async handlers in sync loopguard raise TypeError."""
        async def async_handler(func, args, kwargs):
            return "async result"

        @loopguard(max_repeats=1, on_loop=async_handler)
        def func(x):
            return x

        func(1)

        with pytest.raises(TypeError, match="Use async_loopguard for async handlers"):
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
        """Test that unhashable arguments (like dicts, lists) work correctly."""
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

        assert len(func.get_signatures()) == 150

        # Reset should clear everything including call_count
        func.reset()

        # Verify state is actually cleared
        assert func.get_signatures() == []
        assert func.get_count((0,)) == 0

        # Should work fine after reset
        for i in range(150):
            func(i)

        assert len(func.get_signatures()) == 150


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

    @pytest.mark.asyncio
    async def test_async_reset(self):
        """Test that reset() works correctly for async_loopguard."""
        @async_loopguard(max_repeats=2, window=60)
        async def func(x):
            return x

        await func(1)
        await func(1)
        assert func.get_count((1,)) == 2
        assert func.would_trigger((1,))

        func.reset()

        # Verify state is cleared
        assert func.get_signatures() == []
        assert func.get_count((1,)) == 0
        assert not func.would_trigger((1,))

        # Should work again
        assert await func(1) == 1


class TestLoopDetectedError:
    def test_error_message(self):
        err = LoopDetectedError("my_func", 5, 120)
        msg = str(err)
        assert "my_func" in msg
        assert "5" in msg
        assert "120" in msg
        assert "already called" in msg

    def test_error_attributes(self):
        err = LoopDetectedError("test", 3, 60)
        assert err.func_name == "test"
        assert err.count == 3
        assert err.window == 60

    def test_error_repr(self):
        err = LoopDetectedError("my_func", 3, 60.5)
        r = repr(err)
        assert "my_func" in r
        assert "3" in r
        assert "60.5" in r

    def test_error_equality(self):
        e1 = LoopDetectedError("func", 3, 60.0)
        e2 = LoopDetectedError("func", 3, 60.0)
        e3 = LoopDetectedError("other", 3, 60.0)
        e4 = LoopDetectedError("func", 5, 60.0)

        assert e1 == e2
        assert e1 != e3
        assert e1 != e4
        assert hash(e1) == hash(e2)

    def test_error_equality_not_implemented(self):
        err = LoopDetectedError("func", 3, 60.0)
        assert err != "not an error"
        assert err != 42


class TestWouldTrigger:
    def test_would_trigger_false_initially(self):
        @loopguard(max_repeats=3, window=60)
        def func(x):
            return x

        assert not func.would_trigger((1,))

    def test_would_trigger_after_calls(self):
        @loopguard(max_repeats=2, window=60)
        def func(x):
            return x

        func(1)
        assert not func.would_trigger((1,))
        func(1)
        assert func.would_trigger((1,))

    def test_would_trigger_different_args(self):
        @loopguard(max_repeats=2, window=60)
        def func(x):
            return x

        func(1)
        func(1)
        assert func.would_trigger((1,))
        assert not func.would_trigger((2,))

    @pytest.mark.asyncio
    async def test_async_would_trigger(self):
        @async_loopguard(max_repeats=2, window=60)
        async def func(x):
            return x

        assert not func.would_trigger((1,))
        await func(1)
        await func(1)
        assert func.would_trigger((1,))
