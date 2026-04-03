"""
Streamlit UI for RAG MCP Project
Modern glassmorphism design with gradient accents
"""

import os
import time
import html
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────
# CSS: Glassmorphism + Gradient theme
# ─────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Background ─── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ─── Hide default header ─── */
#MainMenu, header, footer { visibility: hidden; }

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 6px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* ─── Metric cards ─── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 20px 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(102,126,234,0.3);
}
[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ─── Glass card ─── */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 16px;
}

/* ─── Section title ─── */
.section-title {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.section-subtitle {
    color: rgba(255,255,255,0.5);
    font-size: 14px;
    margin-bottom: 24px;
}

/* ─── Chat messages ─── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 8px !important;
}

/* ─── Chat input ─── */
[data-testid="stChatInputContainer"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stChatInputContainer"] textarea {
    color: white !important;
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 24px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.6);
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: none;
}

/* ─── Expanders ─── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* ─── Text inputs ─── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ─── Selectbox ─── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ─── Info / Success / Warning ─── */
.stAlert {
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
}

/* ─── Source badge ─── */
.source-badge {
    display: inline-block;
    background: rgba(102,126,234,0.25);
    border: 1px solid rgba(102,126,234,0.5);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 12px;
    color: #a78bfa;
    margin: 2px 4px 2px 0;
}

/* ─── Status dot ─── */
.status-dot-online { color: #4ade80; font-size: 20px; }
.status-dot-offline { color: #f87171; font-size: 20px; }

/* ─── Progress bar override ─── */
.stProgress > div > div > div {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
}

/* ─── Divider ─── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* ─── Text ─── */
p, li, label, span {
    color: rgba(255,255,255,0.85) !important;
}
h1, h2, h3 {
    color: white !important;
}
</style>
"""


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────

def api_get(path: str, base: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{base}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, base: str, payload: Optional[Dict] = None) -> Optional[Dict]:
    try:
        r = requests.post(f"{base}{path}", json=payload or {}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_delete(path: str, base: str) -> bool:
    try:
        r = requests.delete(f"{base}{path}", timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False


def check_api_status(base: str) -> bool:
    try:
        r = requests.get(f"{base}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def escape_html_text(value: Any) -> str:
    """Escape dynamic text before injecting it into HTML snippets."""
    return html.escape(str(value), quote=True)


# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────

def page_dashboard(api_url: str) -> None:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">System metrics and status at a glance</div>', unsafe_allow_html=True)

    health = api_get("/health", api_url) or {}
    cache = api_get("/cache/stats", api_url) or {}
    docs = api_get("/documents", api_url) or []
    online = health.get("system_ready", False)

    # ─── Status banner ───
    if online:
        st.markdown(
            '<div class="glass-card" style="border-left: 4px solid #4ade80;">'
            '<span class="status-dot-online">●</span> '
            '<strong style="color:white"> System Online</strong> '
            '<span style="color:rgba(255,255,255,0.5);font-size:13px">  API is running and ready to serve requests</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="glass-card" style="border-left: 4px solid #f87171;">'
            '<span class="status-dot-offline">●</span> '
            '<strong style="color:white"> System Offline</strong> '
            '<span style="color:rgba(255,255,255,0.5);font-size:13px">  Could not reach the API</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # ─── Metrics ───
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Queries", cache.get("total_queries", 0))
    with c2:
        st.metric("Cache Hit Rate", cache.get("hit_rate", "0%"))
    with c3:
        st.metric("Cached Entries", cache.get("cached_entries", 0))
    with c4:
        st.metric("Documents", len(docs) if isinstance(docs, list) else health.get("documents_indexed", 0))

    st.write("")

    # ─── Two-column details ───
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Cache Statistics**")
        hits = cache.get("cache_hits", 0)
        misses = cache.get("cache_misses", 0)
        total = cache.get("total_queries", 1)
        if total > 0:
            st.progress(hits / max(total, 1), text=f"Hits: {hits} / Total: {total}")
        st.write(f"Evictions: `{cache.get('evictions', 0)}`")
        st.write(f"TTL: `{cache.get('ttl_minutes', 60)} min`")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**System Info**")
        st.write(f"API URL: `{api_url}`")
        st.write(f"Status: `{health.get('status', 'unknown')}`")
        st.write(f"Cache entries: `{health.get('cache_entries', 0)}`")
        st.write(f"Documents indexed: `{health.get('documents_indexed', 0)}`")
        st.markdown('</div>', unsafe_allow_html=True)

    # ─── Documents preview ───
    st.write("")
    st.markdown("**Indexed Documents**")
    if isinstance(docs, list) and docs:
        cols = st.columns(min(len(docs), 3))
        for i, doc in enumerate(docs):
            with cols[i % 3]:
                size_kb = round(doc.get("size_bytes", 0) / 1024, 1)
                doc_name = escape_html_text(doc.get("name", ""))
                st.markdown(
                    f'<div class="glass-card" style="text-align:center;">'
                    f'<div style="font-size:32px">📄</div>'
                    f'<div style="font-weight:600;color:white;font-size:13px;margin:8px 0">{doc_name}</div>'
                    f'<div style="color:rgba(255,255,255,0.5);font-size:12px">{size_kb} KB</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No documents indexed yet.")


# ─────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────

def init_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def page_chat(api_url: str) -> None:
    init_chat()
    st.markdown('<div class="section-title">Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Ask questions about your indexed documents</div>', unsafe_allow_html=True)

    # ─── Clear button ───
    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ─── Welcome message ───
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:40px;">'
            '<div style="font-size:48px">🤖</div>'
            '<div style="font-size:20px;font-weight:600;color:white;margin:12px 0">RAG Assistant</div>'
            '<div style="color:rgba(255,255,255,0.5);font-size:14px">Ask me anything about your documents</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ─── History ───
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.write(entry["content"])
        else:
            with st.chat_message("assistant"):
                st.write(entry["content"])
                if entry.get("sources"):
                    src_html = "".join(
                        f'<span class="source-badge">📄 {escape_html_text(s)}</span>'
                        for s in entry["sources"]
                    )
                    st.markdown(f'<div style="margin-top:8px">{src_html}</div>', unsafe_allow_html=True)
                cached_label = "⚡ Cached" if entry.get("cached") else "🔍 Live"
                st.caption(f"{cached_label} · {entry.get('latency', 0):.3f}s")

    # ─── Input ───
    question = st.chat_input("Ask anything about your documents...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": [], "cached": False, "latency": 0})
        with st.spinner("Thinking..."):
            t0 = time.time()
            resp = api_post("/query", api_url, {"question": question})
            latency = round(time.time() - t0, 3)
            if resp:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": resp.get("answer", "No answer returned."),
                    "sources": resp.get("sources", []),
                    "cached": resp.get("cached", False),
                    "latency": latency,
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Error: could not reach the API. Is the server running?",
                    "sources": [],
                    "cached": False,
                    "latency": latency,
                })
        st.rerun()


# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────

def page_documents(api_url: str) -> None:
    st.markdown('<div class="section-title">Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Manage your indexed knowledge base</div>', unsafe_allow_html=True)

    docs = api_get("/documents", api_url)

    if docs is None:
        st.error("Could not reach the API. Make sure the server is running.")
        return

    col_info, col_action = st.columns([3, 1])
    with col_info:
        st.markdown(
            f'<div class="glass-card">'
            f'<span style="font-size:28px;font-weight:700;color:white">{len(docs)}</span> '
            f'<span style="color:rgba(255,255,255,0.5)"> documents indexed</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_action:
        if st.button("🔄 Reindex All", type="primary"):
            with st.spinner("Reindexing..."):
                resp = api_post("/documents/reindex", api_url)
                if resp:
                    st.success(resp.get("message", "Done."))

    st.write("")

    if not docs:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:40px;">'
            '<div style="font-size:48px">📂</div>'
            '<div style="color:rgba(255,255,255,0.5);margin-top:12px">No documents found.<br>Place .txt files in the <code>documents/</code> folder and restart the API.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for doc in docs:
        size_kb = round(doc.get("size_bytes", 0) / 1024, 1)
        indexed = doc.get("indexed", False)
        with st.expander(f"📄  {doc.get('name', 'document')}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Size", f"{size_kb} KB")
            c2.metric("Indexed", "✅ Yes" if indexed else "❌ No")
            c3.metric("Format", doc.get("name", "").split(".")[-1].upper())


# ─────────────────────────────────────────────
# Cache Manager
# ─────────────────────────────────────────────

def page_cache(api_url: str) -> None:
    st.markdown('<div class="section-title">Cache Manager</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Monitor and control the query cache</div>', unsafe_allow_html=True)

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    stats = api_get("/cache/stats", api_url)

    if not stats:
        st.error("Could not load cache statistics.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Queries", stats.get("total_queries", 0))
    with c2:
        st.metric("Cache Hits", stats.get("cache_hits", 0))
    with c3:
        st.metric("Cache Misses", stats.get("cache_misses", 0))

    st.write("")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Hit Rate", stats.get("hit_rate", "0%"))
    with c5:
        st.metric("Cached Entries", stats.get("cached_entries", 0))
    with c6:
        st.metric("TTL (min)", stats.get("ttl_minutes", 60))

    st.write("")

    # Visual hit/miss bar
    total = stats.get("total_queries", 1)
    hits = stats.get("cache_hits", 0)
    if total > 0:
        st.markdown("**Cache Efficiency**")
        st.progress(hits / max(total, 1), text=f"{stats.get('hit_rate','0%')} hit rate ({hits}/{total})")

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Details**")
    st.write(f"Evictions: `{stats.get('evictions', 0)}`")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🗑️ Clear All Cache", type="primary"):
        with st.spinner("Clearing..."):
            if api_delete("/cache/clear", api_url):
                st.success("Cache cleared successfully.")
                time.sleep(1)
                st.rerun()


# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────

def page_settings(api_url: str) -> str:
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Configure the connection and preferences</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**API Connection**")
    new_url = st.text_input("API Base URL", value=api_url, help="e.g. http://localhost:8000")

    online = check_api_status(new_url)
    if online:
        st.success("✅  Connected to API")
    else:
        st.error("❌  Cannot reach API — is the server running?")

    if st.button("Save", type="primary"):
        st.session_state.api_url = new_url.strip()
        st.success("Saved.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**About**")
    st.write("**RAG MCP Project** — v1.0.0")
    st.write("Vector store: ChromaDB · Embeddings: sentence-transformers · LLM: Gemini / Ollama")
    st.markdown('</div>', unsafe_allow_html=True)

    return new_url.strip()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="RAG MCP Project",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL

    # ─── App header ───
    st.markdown(
        '<div style="padding: 24px 0 8px 0;">'
        '<span style="font-size:36px;font-weight:800;background:linear-gradient(135deg,#667eea,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">RAG MCP Project</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ─── Tabs ───
    tabs = st.tabs(["📊  Dashboard", "💬  Chat", "📁  Documents", "🗂️  Cache", "⚙️  Settings"])

    with tabs[0]:
        page_dashboard(st.session_state.api_url)
    with tabs[1]:
        page_chat(st.session_state.api_url)
    with tabs[2]:
        page_documents(st.session_state.api_url)
    with tabs[3]:
        page_cache(st.session_state.api_url)
    with tabs[4]:
        page_settings(st.session_state.api_url)


if __name__ == "__main__":
    main()
