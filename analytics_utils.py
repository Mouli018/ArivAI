"""
analytics_utils.py – Lightweight per-query metrics tracking.

Metrics collected per query (no ground truth needed):
  - latency_ms        : wall-clock time from query → full response
  - avg_rerank_score  : mean cross-encoder score of retrieved chunks
  - faithfulness_proxy: keyword overlap between answer and context
  - source            : 'doc' | 'web' | 'general'
  - feedback          : None | 'up' | 'down'  (user thumbs)
  - chunks_retrieved  : how many chunks were used
  - failed_retrieval  : True if we fell back to web/general from doc mode
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional


# ─── Metric logging ───────────────────────────────────────────────────────────

def compute_faithfulness_proxy(answer: str, context: str) -> float:
    """
    Rough faithfulness score: fraction of meaningful answer words found in context.

    Not a replacement for Ragas — this is a fast heuristic.
    Score of 1.0 means every non-stopword in the answer appears in the context.
    Score of 0.0 means the answer shares nothing with the retrieved context.
    """
    if not context.strip() or not answer.strip():
        return 0.0

    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "of",
        "and", "or", "for", "with", "this", "that", "are", "was",
        "be", "by", "as", "i", "you", "we", "they", "he", "she",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "not", "no", "but", "from",
        "your", "my", "our", "its", "which", "who", "what", "when",
        "where", "how", "if", "so", "then", "also", "just", "can",
    }

    def keywords(text: str) -> set:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return {t for t in tokens if t not in stopwords}

    answer_kw = keywords(answer)
    if not answer_kw:
        return 0.0
    context_kw = keywords(context)
    overlap = answer_kw & context_kw
    return round(len(overlap) / len(answer_kw), 3)


def log_query(
    session_state,
    *,
    query: str,
    answer: str,
    source: str,
    latency_ms: float,
    chunks: list,
    context: str,
    failed_retrieval: bool = False,
) -> dict:
    """
    Build a metrics record for one query and append it to session_state.metrics.

    Returns the record dict (so the caller can attach feedback later).
    """
    # Average cross-encoder rerank score from chunks
    scores = [
        c.metadata.get("rerank_score", None)
        for c in chunks
        if hasattr(c, "metadata")
    ]
    valid_scores = [s for s in scores if s is not None]
    avg_rerank = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    faithfulness = compute_faithfulness_proxy(answer, context) if context.strip() else None

    record = {
        "timestamp":        datetime.now().strftime("%H:%M:%S"),
        "query":            query[:80],
        "source":           source,           # 'doc' | 'web' | 'general'
        "latency_ms":       round(latency_ms),
        "chunks_retrieved": len(chunks),
        "avg_rerank_score": avg_rerank,
        "faithfulness":     faithfulness,
        "failed_retrieval": failed_retrieval,
        "feedback":         None,             # set later via thumbs
    }

    if "metrics" not in session_state:
        session_state["metrics"] = []
    session_state["metrics"].append(record)
    return record


# ─── Analytics rendering ──────────────────────────────────────────────────────

def render_analytics_panel(session_state) -> None:
    """
    Render the 📊 Analytics expander inside the Streamlit sidebar.
    Call this from within a `with st.sidebar:` block.
    """
    import streamlit as st

    metrics: List[dict] = session_state.get("metrics", [])

    with st.expander("📊 Analytics", expanded=False):
        if not metrics:
            st.caption("No queries yet — ask something first.")
            return

        n = len(metrics)

        # ── Summary row ──
        avg_lat  = sum(m["latency_ms"] for m in metrics) / n
        n_doc    = sum(1 for m in metrics if m["source"] == "doc")
        n_web    = sum(m["source"] == "web" for m in metrics)
        n_gen    = sum(m["source"] == "general" for m in metrics)
        n_failed = sum(m["failed_retrieval"] for m in metrics)
        thumbs_up   = sum(m["feedback"] == "up"   for m in metrics)
        thumbs_down = sum(m["feedback"] == "down" for m in metrics)

        c1, c2 = st.columns(2)
        c1.metric("⏱ Avg latency", f"{avg_lat:.0f} ms")
        c2.metric("📝 Total queries", n)

        c3, c4 = st.columns(2)
        c3.metric("👍 Positive", thumbs_up)
        c4.metric("👎 Negative", thumbs_down)

        st.caption("**Source distribution**")
        src_cols = st.columns(3)
        src_cols[0].metric("📄 Doc",     n_doc)
        src_cols[1].metric("🌐 Web",     n_web)
        src_cols[2].metric("💬 General", n_gen)

        if n_failed:
            st.warning(f"⚠️ {n_failed} failed doc retrieval(s) → fell back to web/general")

        # ── Avg rerank score ──
        rr_scores = [m["avg_rerank_score"] for m in metrics if m["avg_rerank_score"] is not None]
        if rr_scores:
            avg_rr = sum(rr_scores) / len(rr_scores)
            st.metric("🎯 Avg context quality", f"{avg_rr:.3f}")

        # ── Avg faithfulness ──
        faith_scores = [m["faithfulness"] for m in metrics if m["faithfulness"] is not None]
        if faith_scores:
            avg_f = sum(faith_scores) / len(faith_scores)
            st.metric("🔗 Avg faithfulness proxy", f"{avg_f:.2%}")

        # ── Per-query table (last 10) ──
        st.caption("**Recent queries**")
        for m in reversed(metrics[-10:]):
            lat_color = "🟢" if m["latency_ms"] < 3000 else "🟡" if m["latency_ms"] < 7000 else "🔴"
            fb = {"up": "👍", "down": "👎", None: "—"}[m["feedback"]]
            st.markdown(
                f"`{m['timestamp']}` {lat_color} **{m['latency_ms']} ms** "
                f"· {m['source']} · {fb} · _{m['query']}_"
            )
