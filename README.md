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

### Async Support

```python
from loopguard import async_loopguard

@async_loopguard(max_repeats=3, window=60)
async def async_agent_action(query: str) -> str:
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

## API

### `loopguard(max_repeats=3, window=60, on_loop=None)`

Decorator for sync functions.

- `max_repeats`: Max calls with identical args within window (default: 3)
- `window`: Time window in seconds (default: 60)
- `on_loop`: Optional callback `(func, args, kwargs) -> Any`. If provided, return value is used instead of raising.

### `async_loopguard(max_repeats=3, window=60, on_loop=None)`

Same as above, for async functions.

### `LoopDetectedError`

Raised when loop detected (unless `on_loop` provided).

Attributes:
- `func_name`: Name of the looping function
- `count`: Number of repeated calls
- `window`: Time window in seconds

## How It Works

1. Hash function arguments to create a signature
2. Track call timestamps per signature
3. Clean entries outside the time window
4. If calls with same signature exceed `max_repeats`, trigger loop handler

## License

MIT
