# LoopGuard

Protect AI agents from infinite loops. One decorator, zero dependencies.

## The Problem

AI agents get stuck in loops. They call the same function with the same arguments over and over, burning tokens and failing silently. This happens in production more than you'd think ([50% of n8n users report loops](https://community.n8n.io)).

## The Solution

```python
from loopguard import loopguard

@loopguard(max_repeats=3, window=60)
def agent_action(query: str) -> str:
    return llm.complete(query)

# Third call with same query within 60s raises LoopDetectedError
```

## Installation

```bash
pip install loopguard
```

## Features

- **Thread-safe** - Safe for multi-threaded applications
- **Memory-safe** - Auto-cleans old signatures, no memory leaks
- **Zero dependencies** - Only Python stdlib
- **Async support** - Works with async/await
- **Non-blocking handlers** - Custom handlers don't block other calls
- **Type hints** - Full typing support with `py.typed`
- **Sub-second precision** - Float windows like `window=0.5` for rate limiting
- **Clock-immune** - Uses monotonic time, immune to system clock changes

## Usage

### Basic

```python
from loopguard import loopguard, LoopDetectedError

@loopguard(max_repeats=3, window=60)
def search(query: str) -> str:
    return search_api.search(query)

try:
    for _ in range(10):
        search("same query")  # Raises on 4th call
except LoopDetectedError as e:
    print(f"Loop stopped: {e}")
```

### Custom Handler

```python
@loopguard(max_repeats=3, on_loop=lambda f, a, k: "Loop detected, stopping")
def agent_step(state: dict) -> str:
    return llm.complete(state["query"])
```

### Check Call Count

```python
@loopguard(max_repeats=5, window=60)
def my_func(x):
    return x

my_func(10)
my_func(10)
print(my_func.get_count((10,)))  # 2
```

### Reset History

```python
@loopguard(max_repeats=2, window=60)
def my_func(x):
    return x

my_func(5)
my_func(5)
my_func.reset()  # Clear all history
my_func(5)  # Works again
```

### Async Support

```python
from loopguard import async_loopguard

@async_loopguard(max_repeats=3, window=60)
async def async_agent_action(query: str) -> str:
    return await llm.complete(query)
```

### Async with Async Handler

```python
async def my_handler(func, args, kwargs):
    await log_loop_event()
    return "fallback response"

@async_loopguard(max_repeats=3, on_loop=my_handler)
async def agent_action(query: str) -> str:
    return await llm.complete(query)
```

### With LangChain

```python
from langchain.tools import tool
from loopguard import loopguard

@tool
@loopguard(max_repeats=3, window=120)
def search_tool(query: str) -> str:
    """Search the web."""
    return search_api.search(query)
```

### With CrewAI

```python
from crewai import Agent, Task
from loopguard import loopguard

@loopguard(max_repeats=5, window=300)
def execute_task(task: Task) -> str:
    return agent.execute(task)
```

### Sub-second Rate Limiting

```python
# Allow max 5 calls per 500ms
@loopguard(max_repeats=5, window=0.5)
def rate_limited_api(query: str) -> str:
    return api.call(query)
```

## API

### `loopguard(max_repeats=3, window=60, on_loop=None)`

Decorator for sync functions. Thread-safe.

- `max_repeats`: Max calls with identical args within window (default: 3)
- `window`: Time window in seconds, can be float for sub-second precision (default: 60)
- `on_loop`: Optional callback `(func, args, kwargs) -> Any`. If provided, return value is used instead of raising.

Uses monotonic time internally, so immune to system clock adjustments.

**Attached methods:**
- `func.reset()` - Clear all call history
- `func.get_count(args, kwargs)` - Get current count for specific arguments
- `func.would_trigger(args, kwargs)` - Check if next call would trigger loop detection

### `async_loopguard(max_repeats=3, window=60, on_loop=None)`

Same as above, for async functions. Coroutine-safe.

The `on_loop` callback can be sync or async - both are handled correctly.

### `LoopDetectedError`

Raised when loop detected (unless `on_loop` provided).

Attributes:
- `func_name`: Name of the looping function
- `count`: Number of repeated calls that triggered detection
- `window`: Time window in seconds

## How It Works

1. Hash function arguments to create a signature (SHA-256, truncated)
2. Track call timestamps per signature (thread-safe)
3. Clean entries outside the time window
4. If calls with same signature exceed `max_repeats`, trigger loop handler
5. Periodically clean old signatures to prevent memory growth

## Thread Safety

Both `loopguard` and `async_loopguard` are safe for concurrent use:

```python
import threading

@loopguard(max_repeats=100, window=60)
def my_func(x):
    return x

# Safe to call from multiple threads
threads = [threading.Thread(target=lambda: my_func(1)) for _ in range(10)]
```

## License

MIT
