# Custom Scripts

Place your custom Python agent scripts here.
They are mounted into the container at `/app/src` and hot-reloaded.

## Example: Custom tool

```python
# src/my_tool.py
from langchain_core.tools import tool

@tool
def search_company_db(query: str) -> str:
    """Search the internal company database."""
    # your implementation here
    return "..."
```

The FastAPI server watches this directory and reloads automatically.
