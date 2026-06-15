"""
llm_utils.py – LLM (Groq/LLaMA-4) and Tavily web-search helpers.
"""

import json
import os
from typing import Generator, List, Optional, Tuple

import requests

from config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    GROQ_URL,
    MODEL_NAME,
    RECENT_KEYWORDS,
)

# ─── API key accessors (read lazily so Streamlit secrets inject first) ────────

def _groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")

def _tavily_key() -> str:
    return os.environ.get("TAVILY_API_KEY", "")


# ─── LLaMA-4 via Groq ─────────────────────────────────────────────────────────

# Grounded system prompt — prevents hallucination when context is provided
_SYSTEM_WITH_CONTEXT = (
    "You are a precise and trustworthy document Q&A assistant. "
    "Answer ONLY using the information in the provided context passages. "
    "If the answer is not explicitly stated in the context, respond with: "
    "'I could not find this information in the provided documents.' "
    "Never fabricate facts, statistics, or details not present in the context. "
    "When you use information from the context, mention the source document and page if available. "
    "Format your response using clear Markdown."
)

# General system prompt — used when no document context is provided
_SYSTEM_GENERAL = (
    "You are a helpful and concise AI assistant. "
    "Answer the question as accurately as possible. "
    "If you are not confident in your answer, say so clearly. "
    "Format responses using Markdown."
)


def query_llama(
    query: str,
    context: str = "",
    sources: Optional[List[Tuple[str, int]]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Query LLaMA-4 Scout via the Groq API.

    Args:
        query:       The user's question.
        context:     Optional retrieved document context to ground the answer.
        sources:     Optional list of (filename, page) citation tuples.
        max_tokens:  Maximum tokens for the response.
        temperature: Sampling temperature.

    Returns:
        The model's response, or a descriptive error message.
    """
    api_key = _groq_key()
    if not api_key:
        return (
            "❌ **GROQ_API_KEY is not set.**\n\n"
            "Add it to your `.env` file locally, or to Streamlit Cloud secrets."
        )

    has_context = bool(context.strip())
    system_prompt = _SYSTEM_WITH_CONTEXT if has_context else _SYSTEM_GENERAL

    if has_context:
        # Build source hint string if citations available
        source_hint = ""
        if sources:
            refs = "; ".join(
                f"{src} p.{page}" for src, page in sources
            )
            source_hint = f"\n\n**Sources:** {refs}"
        user_content = (
            f"**Context passages:**\n{context}{source_hint}"
            f"\n\n**Question:** {query}"
        )
    else:
        user_content = f"**Question:** {query}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as exc:
        status = resp.status_code
        if status == 401:
            return "❌ Invalid GROQ_API_KEY. Please check your credentials."
        if status == 429:
            return "⏳ Rate limit reached. Please wait a moment and try again."
        return f"❌ Groq API error ({status}): {exc}"
    except Exception as exc:
        return f"❌ Unexpected error calling LLaMA: {exc}"


def query_llama_stream(
    query: str,
    context: str = "",
    sources: Optional[List[Tuple[str, int]]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Generator[str, None, None]:
    """
    Stream LLaMA-4 response token-by-token via Groq SSE.

    Yields individual text chunks as they arrive so Streamlit can
    display them with a real-time typing effect (use with st.write_stream).
    Yields a single error string on failure.
    """
    api_key = _groq_key()
    if not api_key:
        yield "❌ **GROQ_API_KEY is not set.** Add it to your `.env` file."
        return

    has_context = bool(context.strip())
    system_prompt = _SYSTEM_WITH_CONTEXT if has_context else _SYSTEM_GENERAL

    if has_context:
        source_hint = ""
        if sources:
            refs = "; ".join(f"{src} p.{page}" for src, page in sources)
            source_hint = f"\n\n**Sources:** {refs}"
        user_content = (
            f"**Context passages:**\n{context}{source_hint}"
            f"\n\n**Question:** {query}"
        )
    else:
        user_content = f"**Question:** {query}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        with requests.post(
            GROQ_URL, headers=headers, json=payload,
            stream=True, timeout=60
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                # SSE format: "data: {...}" or "data: [DONE]"
                if raw_line.startswith(b"data: "):
                    data = raw_line[6:]
                    if data.strip() == b"[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue
    except requests.exceptions.Timeout:
        yield "\n\n⏱️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as exc:
        status = resp.status_code
        if status == 429:
            yield "\n\n⏳ Rate limit reached. Please wait a moment."
        else:
            yield f"\n\n❌ Groq API error ({status}): {exc}"
    except Exception as exc:
        yield f"\n\n❌ Unexpected error: {exc}"


# ─── Tavily web search fallback ───────────────────────────────────────────────

def tavily_fallback(query: str) -> str:
    """
    Fetch a synthesised answer from the Tavily Search API.

    Args:
        query: The user's question.

    Returns:
        A concise answer string, or a descriptive error message.
    """
    api_key = _tavily_key()
    if not api_key:
        return (
            "❌ **TAVILY_API_KEY is not set.**\n\n"
            "Web search is unavailable. Add it to your `.env` file or Streamlit Cloud secrets."
        )

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 5,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        answer: Optional[str] = resp.json().get("answer")
        return answer if answer else "No relevant information found on the web."
    except requests.exceptions.Timeout:
        return "⏱️ Web search timed out. Please try again."
    except requests.exceptions.HTTPError as exc:
        return f"❌ Tavily API error: {exc}"
    except Exception as exc:
        return f"❌ Web search error: {exc}"


# ─── Recent-query detection ───────────────────────────────────────────────────

def is_recent_query(query: str) -> bool:
    """Return True if the query likely needs up-to-date web information."""
    q = query.lower()
    return any(kw in q for kw in RECENT_KEYWORDS)
