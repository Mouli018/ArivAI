"""
streamlit_app.py – Premium dark-themed Streamlit UI for the LLaMA-4 RAG Chatbot.

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push to GitHub → connect repo on share.streamlit.io → add secrets.
"""

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict

import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="LLaMA-4 RAG Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Streamlit Cloud secrets into env vars before any other import ──────
def _init_secrets() -> None:
    for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "MEM0_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
        try:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = st.secrets[key]
        except Exception:
            pass  # Running locally – .env is loaded by config.py via python-dotenv

_init_secrets()

# ── Application imports (after secrets are injected) ─────────────────────────
import time
from config import DOCS_PATH, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, SUPPORTED_EXTENSIONS
from document_utils import (
    build_all_vectorstores, create_vectorstore, load_documents_grouped,
    hybrid_search, rerank_chunks, build_citation_text, VectorstoreBundle,
)
from llm_utils import is_recent_query, query_llama, query_llama_stream, tavily_fallback
from memory_utils import add_memory, search_memory, clear_memory, memory_enabled
from analytics_utils import log_query, render_analytics_panel
from guardrails import evaluate_guardrails, redact_pii

# ─────────────────────────────────────────────────────────────────────────────
# CSS – Premium dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #12103a 55%, #0c1f3d 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 8, 30, 0.97) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.25) !important;
}

[data-testid="stSidebarContent"] { padding-top: 1.5rem; }

/* ── Main header ── */
.main-header {
    text-align: center;
    padding: 1.2rem 0 0.5rem;
}
.main-header h1 {
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 16px !important;
    margin-bottom: 10px !important;
    padding: 4px 8px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(124, 58, 237, 0.25) !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(124, 58, 237, 0.08) !important;
    border-color: rgba(124, 58, 237, 0.25) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(124, 58, 237, 0.35) !important;
    color: #e2e8f0 !important;
    border-radius: 16px !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .stMarkdown p { color: #94a3b8 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(124, 58, 237, 0.04) !important;
    border: 1.5px dashed rgba(124, 58, 237, 0.35) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

/* ── Source badge ── */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.28);
    color: #38bdf8;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ── Memory badge ── */
.memory-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.35);
    color: #c084fc;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 4px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    text-align: center;
    gap: 1.2rem;
}
.welcome-icon { font-size: 4rem; animation: float 3s ease-in-out infinite; }
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-10px); }
}
.welcome-title {
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.welcome-sub { color: #64748b; font-size: 1rem; max-width: 480px; line-height: 1.6; }
.welcome-hints {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin-top: 0.5rem;
}
.hint-chip {
    background: rgba(124, 58, 237, 0.1);
    border: 1px solid rgba(124, 58, 237, 0.25);
    color: #c4b5fd;
    font-size: 0.8rem;
    padding: 6px 14px;
    border-radius: 20px;
    cursor: default;
}

/* ── Dividers & text ── */
hr { border-color: rgba(255,255,255,0.07) !important; }
.stMarkdown p, .stMarkdown li { color: #cbd5e1; }
.stMarkdown h2, .stMarkdown h3 { color: #e2e8f0; }

/* ── Sidebar section labels ── */
.sidebar-label {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
    margin-top: 16px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.4); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.65); }

/* ── Hide Streamlit chrome but keep sidebar toggle ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar collapse/expand toggle ── */
/* Try all known Streamlit selectors across versions */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="borderless"][data-testid="baseButton-borderless"] {
    visibility: visible !important;
    opacity: 1 !important;
    display: flex !important;
}
[data-testid="stSidebarCollapsedControl"] {
    background: rgba(124, 58, 237, 0.2) !important;
    border: 1px solid rgba(124, 58, 237, 0.5) !important;
    border-radius: 0 8px 8px 0 !important;
    top: 0.8rem !important;
    left: 0 !important;
    position: fixed !important;
    z-index: 9999 !important;
    padding: 8px 6px !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #a78bfa !important;
    color: #a78bfa !important;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    background: rgba(124, 58, 237, 0.45) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],          # list[dict]: role, content, source, citations, msg_id
        "selected_doc": None,    # str | None
        "extra_vs": {},          # filename -> VectorstoreBundle (uploaded files)
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "metrics": [],           # list[dict]: per-query analytics records
        "_msg_counter": 0,       # unique ID for each message
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Cached vectorstore loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_vectorstores() -> Dict:
    """Build/load Qdrant vectorstores from the docs/ folder. Cached for the session."""
    return build_all_vectorstores(DOCS_PATH)

# Load vectorstores BEFORE sidebar — show spinner in main area so sidebar isn't blocked
if "_vs_loaded" not in st.session_state:
    with st.spinner("Loading documents into Qdrant... (first load only, please wait)"):
        _base_vs_cache = _load_vectorstores()
    st.session_state["_vs_loaded"] = True
else:
    _base_vs_cache = _load_vectorstores()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h1 style='margin-bottom:4px;'>🦙 LLaMA-4 RAG</h1>"
        "<p style='color:#64748b;font-size:0.8rem;margin-top:0;'>Powered by Groq · FAISS · Tavily</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Stats ──
    base_vs = _load_vectorstores()   # already cached — instant
    all_vs: Dict = {**base_vs, **st.session_state.extra_vs}
    total_docs = len(all_vs)

    col1, col2 = st.columns(2)
    col1.metric("📄 Docs", total_docs)
    col2.metric("💬 Chats", len(st.session_state.messages) // 2)

    # ── Analytics panel ──
    render_analytics_panel(st.session_state)

    st.markdown("<div class='sidebar-label'>📂 Document</div>", unsafe_allow_html=True)

    doc_choices = ["🌐 General (no doc)"] + sorted(all_vs.keys())
    selected_label = st.selectbox(
        "Select a document",
        doc_choices,
        index=0,
        label_visibility="collapsed",
        key="doc_selector",
    )
    st.session_state.selected_doc = (
        None if selected_label == "🌐 General (no doc)" else selected_label
    )

    # ── Upload ──
    st.markdown("<div class='sidebar-label'>⬆️ Upload document</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
        key="uploader",
    )

    if uploaded is not None and uploaded.name not in all_vs:
        with st.spinner(f"Embedding **{uploaded.name}**…"):
            # Save temp file to docs/ so the loader can read it
            save_path = os.path.join(DOCS_PATH, uploaded.name)
            os.makedirs(DOCS_PATH, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            try:
                grouped = load_documents_grouped(DOCS_PATH)
                if uploaded.name in grouped:
                    vs = create_vectorstore(grouped[uploaded.name])
                    st.session_state.extra_vs[uploaded.name] = vs
                    st.success(f"✅ **{uploaded.name}** is ready!")
                    st.rerun()
            except Exception as exc:
                st.error(f"❌ Failed to embed: {exc}")

    # ── LLM settings ──
    st.markdown("<div class='sidebar-label'>⚙️ LLM settings</div>", unsafe_allow_html=True)
    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        help="Higher → more creative. Lower → more precise.",
    )
    st.session_state.max_tokens = st.slider(
        "Max tokens", 256, 2048, st.session_state.max_tokens, 64,
        help="Maximum length of the model's response.",
    )

    st.divider()

    # ── Memory status ──
    if memory_enabled():
        st.markdown("<div class='sidebar-label'>🧠 Mem0 Memory</div>", unsafe_allow_html=True)
        st.markdown(
            "<span class='memory-badge'>🟢 Active – remembers past chats</span>",
            unsafe_allow_html=True,
        )
        if st.button("🧹 Clear memory", key="clear_memory_btn"):
            if clear_memory():
                st.success("Memory cleared!")
            else:
                st.error("Could not clear memory.")
    else:
        st.markdown("<div class='sidebar-label'>🧠 Memory</div>", unsafe_allow_html=True)
        st.caption("Add MEM0_API_KEY to enable.")

    st.divider()

    # ── Clear chat ──
    if st.button("🗑️ Clear chat", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='color:#334155;font-size:0.72rem;text-align:center;margin-top:1rem;'>"
        "© 2025 LLaMA-4 RAG · MIT License</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main – header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='main-header'>"
    "<h1>🦙 LLaMA-4 RAG Chatbot</h1>"
    "<p>Ask questions from your documents or let the web answer for you.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Chat history display
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class='welcome-wrap'>
            <div class='welcome-icon'>🦙</div>
            <p class='welcome-title'>Ask me anything</p>
            <p class='welcome-sub'>
                Select a document from the sidebar to query your files,
                or just type a question for a general answer.
                Time-sensitive queries are automatically routed to the web.
            </p>
            <div class='welcome-hints'>
                <span class='hint-chip'>📄 Q&amp;A from PDFs</span>
                <span class='hint-chip'>🌐 Live web search</span>
                <span class='hint-chip'>🤖 LLaMA-4 powered</span>
                <span class='hint-chip'>⬆️ Upload your own docs</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🦙"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("source"):
                src = msg["source"]
                icon = "🌐" if src == "web" else "📄"
                label = "Web search (Tavily)" if src == "web" else src
                st.markdown(
                    f"<span class='source-badge'>{icon} {label}</span>",
                    unsafe_allow_html=True,
                )
            # Show citations from history
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📎 View Sources", expanded=False):
                    for cit in msg["citations"]:
                        score = cit.get('rerank_score', '')
                        score_txt = f" &nbsp;·&nbsp; score: `{score:.3f}`" if score else ""
                        st.markdown(
                            f"**{cit['source']}** &nbsp;—&nbsp; page {cit['page']}{score_txt}",
                            unsafe_allow_html=True,
                        )
            # ── Feedback buttons on assistant messages ──
            if msg["role"] == "assistant":
                mid = msg.get("msg_id", i)
                met_idx = msg.get("metrics_idx", None)
                fb_col1, fb_col2, _ = st.columns([1, 1, 10])
                current_fb = st.session_state.metrics[met_idx]["feedback"] if (
                    met_idx is not None
                    and met_idx >= 0
                    and met_idx < len(st.session_state.metrics)
                ) else None
                if fb_col1.button("👍", key=f"up_{mid}", help="Good answer",
                                  type="primary" if current_fb == "up" else "secondary"):
                    if met_idx is not None and met_idx >= 0 and met_idx < len(st.session_state.metrics):
                        st.session_state.metrics[met_idx]["feedback"] = "up"
                    st.rerun()
                if fb_col2.button("👎", key=f"dn_{mid}", help="Bad answer",
                                  type="primary" if current_fb == "down" else "secondary"):
                    if met_idx is not None and met_idx >= 0 and met_idx < len(st.session_state.metrics):
                        st.session_state.metrics[met_idx]["feedback"] = "down"
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Chat input & response logic
# ─────────────────────────────────────────────────────────────────────────────
placeholder = (
    f"Ask about '{st.session_state.selected_doc}' …"
    if st.session_state.selected_doc
    else "Ask anything … (web search activates for recent queries)"
)

if prompt := st.chat_input(placeholder, key="chat_input"):
    # ── Input Guardrails (PII & LLM Security Check) ──
    prompt = redact_pii(prompt)
    is_safe, violation_reason = evaluate_guardrails(prompt)

    # ── Show user message immediately ──
    st.session_state.messages.append({"role": "user", "content": prompt, "source": None})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    if not is_safe:
        warning_msg = f"🚨 **Guardrail Alert:** I cannot fulfill this request as it violates safety policies ({violation_reason})."
        with st.chat_message("assistant", avatar="🦙"):
            st.markdown(warning_msg)
            
        st.session_state["_msg_counter"] += 1
        st.session_state.messages.append(
            {"role": "assistant", "content": warning_msg, "source": "guardrail",
             "citations": [], "msg_id": st.session_state["_msg_counter"], "metrics_idx": -1}
        )
        st.stop()

    # ── Generate response ──
    with st.chat_message("assistant", avatar="🦙"):
        answer = ""
        source_label: str | None = None
        used_memory: bool = False
        used_chunks: list = []
        citations: list = []
        context_for_metrics: str = ""
        failed_retrieval: bool = False
        t_start = time.time()
        all_vs_now: Dict = {**_load_vectorstores(), **st.session_state.extra_vs}

        # ── Retrieve relevant past memories ──
        past_memory_ctx = search_memory(prompt)
        used_memory = bool(past_memory_ctx)

        if is_recent_query(prompt):
            # Route to web search for time-sensitive queries
            with st.spinner("🌐 Searching the web..."):
                answer = tavily_fallback(prompt)
            source_label = "web"

        elif st.session_state.selected_doc and st.session_state.selected_doc in all_vs_now:
            # ── Hybrid retrieval → re-rank ──
            with st.spinner("🔍 Retrieving & ranking chunks..."):
                bundle = all_vs_now[st.session_state.selected_doc]
                candidates = hybrid_search(bundle, prompt)
                used_chunks = rerank_chunks(prompt, candidates)
                citations = build_citation_text(used_chunks)

            top_context = "\n\n".join(c.page_content for c in used_chunks)
            combined_context = "\n\n".join(filter(None, [past_memory_ctx, top_context]))
            context_for_metrics = top_context
            src_tuples = [(c["source"], c["page"]) for c in citations]

            if len(top_context.strip()) > 80:
                # ── STREAM the LLaMA response ──
                answer = st.write_stream(query_llama_stream(
                    prompt,
                    context=combined_context,
                    sources=src_tuples,
                    max_tokens=st.session_state.max_tokens,
                    temperature=st.session_state.temperature,
                ))
                source_label = st.session_state.selected_doc
            else:
                # Weak match → stream general then check for fallback
                failed_retrieval = True
                placeholder = st.empty()
                answer = st.write_stream(query_llama_stream(
                    prompt,
                    context=past_memory_ctx,
                    max_tokens=st.session_state.max_tokens,
                    temperature=st.session_state.temperature,
                ))
                source_label = None
                if not answer.strip() or "i could not find" in answer.lower() or "i don't know" in answer.lower():
                    with st.spinner("🌐 Falling back to web search..."):
                        answer = tavily_fallback(prompt)
                    source_label = "web"
                    st.markdown(answer)

        else:
            # No document selected → stream LLaMA with memory
            answer = st.write_stream(query_llama_stream(
                prompt,
                context=past_memory_ctx,
                max_tokens=st.session_state.max_tokens,
                temperature=st.session_state.temperature,
            ))
            source_label = None
            if not answer.strip() or "i could not find" in answer.lower() or "i don't know" in answer.lower():
                failed_retrieval = True
                with st.spinner("🌐 Falling back to web search..."):
                    answer = tavily_fallback(prompt)
                source_label = "web"
                st.markdown(answer)

        latency_ms = (time.time() - t_start) * 1000

        # ── Save to Mem0 ──
        add_memory(prompt, answer)

        # ── Log metrics ──
        src_label_for_metrics = "web" if source_label == "web" else (
            "doc" if source_label else "general"
        )
        metrics_record = log_query(
            st.session_state,
            query=prompt,
            answer=answer,
            source=src_label_for_metrics,
            latency_ms=latency_ms,
            chunks=used_chunks,
            context=context_for_metrics,
            failed_retrieval=failed_retrieval,
        )
        metrics_idx = len(st.session_state.metrics) - 1

        # ── Badges ──
        if source_label:
            icon = "🌐" if source_label == "web" else "📄"
            label = "Web search (Tavily)" if source_label == "web" else source_label
            st.markdown(f"<span class='source-badge'>{icon} {label}</span>", unsafe_allow_html=True)
        if used_memory:
            st.markdown("<span class='memory-badge'>🧠 memory recalled</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='source-badge' style='background:rgba(30,215,96,.12);border-color:rgba(30,215,96,.3);color:#1ed760;'>⏱ {latency_ms:.0f} ms</span>", unsafe_allow_html=True)

        # ── View Sources ──
        if citations:
            with st.expander("📎 View Sources", expanded=False):
                for cit in citations:
                    score = cit.get('rerank_score', None)
                    score_txt = f" &nbsp;·&nbsp; relevance: `{score:.3f}`" if score is not None else ""
                    st.markdown(f"📄 **{cit['source']}** &nbsp;—&nbsp; page {cit['page']}{score_txt}", unsafe_allow_html=True)

        # ── Feedback buttons ──
        st.session_state["_msg_counter"] += 1
        mid = st.session_state["_msg_counter"]
        fb_col1, fb_col2, _ = st.columns([1, 1, 10])
        if fb_col1.button("👍", key=f"up_{mid}_new", help="Good answer"):
            st.session_state.metrics[metrics_idx]["feedback"] = "up"
            st.rerun()
        if fb_col2.button("👎", key=f"dn_{mid}_new", help="Bad answer"):
            st.session_state.metrics[metrics_idx]["feedback"] = "down"
            st.rerun()

    # ── Persist to session history ──
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "source": source_label,
         "citations": citations, "msg_id": mid, "metrics_idx": metrics_idx}
    )
