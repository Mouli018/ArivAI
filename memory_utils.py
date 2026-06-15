"""
memory_utils.py – Persistent conversation memory via Mem0.

Mem0 stores memories in the cloud, keyed by user_id.
Each chat exchange is added as a memory so future queries
can retrieve relevant past context automatically.
"""

import os
from typing import Optional

import config  # noqa: F401 – ensures .env is loaded

# Default user ID (single-user local app)
DEFAULT_USER_ID = "rag_user"


def _get_client():
    """Return a MemoryClient if MEM0_API_KEY is set, else None."""
    api_key = os.environ.get("MEM0_API_KEY", "")
    if not api_key:
        return None
    try:
        from mem0 import MemoryClient
        return MemoryClient(api_key=api_key)
    except Exception:
        return None


def add_memory(user_message: str, assistant_message: str, user_id: str = DEFAULT_USER_ID) -> None:
    """Store a Q&A exchange in Mem0 memory."""
    client = _get_client()
    if not client:
        return
    try:
        messages = [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        client.add(messages, user_id=user_id)
    except Exception:
        pass  # Memory is best-effort; never crash the app


def search_memory(query: str, user_id: str = DEFAULT_USER_ID, limit: int = 5) -> str:
    """
    Search Mem0 for memories relevant to the query.

    Returns a formatted string of past memories to inject into the LLM prompt,
    or an empty string if none found / Mem0 unavailable.
    """
    client = _get_client()
    if not client:
        return ""
    try:
        results = client.search(query, user_id=user_id, limit=limit)
        if not results:
            return ""
        memories = [r["memory"] for r in results if r.get("memory")]
        if not memories:
            return ""
        formatted = "\n".join(f"- {m}" for m in memories)
        return f"**Relevant memories from past conversations:**\n{formatted}"
    except Exception:
        return ""


def clear_memory(user_id: str = DEFAULT_USER_ID) -> bool:
    """Delete all memories for this user. Returns True on success."""
    client = _get_client()
    if not client:
        return False
    try:
        client.delete_all(user_id=user_id)
        return True
    except Exception:
        return False


def memory_enabled() -> bool:
    """Return True if Mem0 is configured and available."""
    return bool(os.environ.get("MEM0_API_KEY", ""))
