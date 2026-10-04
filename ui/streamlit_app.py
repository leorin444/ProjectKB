"""Streamlit UI for the Local Knowledge Base.

Communicates with the FastAPI backend ONLY via HTTP (X-API-Key auth).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("KB_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("KB_API_KEY", "")

st.set_page_config(page_title="Local Knowledge Base", page_icon="🧠", layout="wide")


def api_headers() -> dict[str, str]:
    return {"X-API-Key": st.session_state.get("api_key", API_KEY)}


def api_get(path: str, **kwargs):
    r = httpx.get(f"{API_URL}{path}", headers=api_headers(), timeout=30.0, **kwargs)
    r.raise_for_status()
    return r.json()


def api_post(path: str, **kwargs):
    r = httpx.post(f"{API_URL}{path}", headers=api_headers(), timeout=600.0, **kwargs)
    r.raise_for_status()
    return r.json()


# ---- Sidebar: API config ----
with st.sidebar:
    st.title("🧠 Knowledge Base")
    st.markdown("---")
    api_key_input = st.text_input(
        "API Key", value=st.session_state.get("api_key", API_KEY), type="password"
    )
    st.session_state["api_key"] = api_key_input

    if st.button("Check Health"):
        try:
            h = httpx.get(f"{API_URL}/health", timeout=5.0).json()
            overall = h.get("status", "unknown")
            color = "green" if overall == "ok" else "orange"
            st.markdown(f"**Status:** :{color}[{overall.upper()}]")
            for name, comp in h.get("components", {}).items():
                icon = "✅" if comp["ok"] else "❌"
                st.caption(f"{icon} {name}: {comp['detail']}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")


# ---- Main Tabs ----
tab_upload, tab_docs, tab_ask = st.tabs(["📁 Upload", "📄 Documents", "💬 Ask"])


# ================ UPLOAD TAB ================
with tab_upload:
    st.header("Upload Document or Screenshot")
    uploaded = st.file_uploader(
        "Choose a file (PDF, PNG, JPG, TXT, MD)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "txt", "md"],
    )
    tags_input = st.text_input("Tags (comma-separated, optional)", placeholder="tutorial, setup")

    if uploaded and st.button("⬆️ Ingest"):
        with st.spinner("Ingesting... (image VLM analysis may take a few minutes on CPU)"):
            tag_list = [t.strip() for t in tags_input.split(",") if t.strip()]
            import json

            try:
                files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
                result = httpx.post(
                    f"{API_URL}/ingest/file",
                    headers=api_headers(),
                    files=files,
                    data={"tags": json.dumps(tag_list)},
                    timeout=600.0,
                ).json()
                if "document_id" in result:
                    st.success(
                        f"✅ Ingested **{result.get('title', uploaded.name)}** "
                        f"({result.get('chunk_count', 0)} chunks)"
                    )
                    st.json(result)
                else:
                    st.error(f"Ingestion error: {result}")
            except Exception as exc:
                st.error(f"Error: {exc}")


# ================ DOCUMENTS TAB ================
with tab_docs:
    st.header("Ingested Documents")
    try:
        docs = api_get("/documents")
        if not docs:
            st.info("No documents ingested yet. Use the Upload tab to add some.")
        else:
            for doc in docs:
                with st.expander(
                    f"📄 {doc['title']} ({doc['type']}) — {doc['created_at'][:10]}"
                ):
                    st.caption(f"ID: `{doc['id']}` | Tags: {', '.join(doc.get('tags', []))}")
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        if st.button("View Markdown", key=f"view_{doc['id']}"):
                            try:
                                full_doc = api_get(f"/documents/{doc['id']}")
                                st.code(
                                    full_doc.get("markdown", "(no content)")[:5000],
                                    language="markdown",
                                )
                            except Exception as e:
                                st.error(str(e))

                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{doc['id']}"):
                            try:
                                api_post(
                                    f"/documents/{doc['id']}",
                                )
                                st.warning("Deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
    except Exception as e:
        st.error(f"Could not fetch documents: {e}")


# ================ ASK TAB ================
with tab_ask:
    st.header("Ask a Question")
    question = st.text_area("Your question", placeholder="How did I fix the peer dependency error?")
    top_k = st.slider("Max chunks to retrieve", 2, 20, 5)

    if st.button("🔍 Ask") and question.strip():
        with st.spinner("Thinking..."):
            try:
                result = api_post("/query", json={"question": question, "top_k": top_k})
                if result.get("found"):
                    st.success("Answer found")
                    st.markdown(result["answer"])
                    st.markdown("---")
                    st.subheader("Citations")
                    for c in result.get("citations", []):
                        title = c.get("title", "Document")
                        section = c.get("section", "General")
                        ts = ""
                        if c.get("start_ts") and c.get("end_ts"):
                            def _fmt(sec):
                                s = int(sec)
                                h, m, s2 = s // 3600, (s % 3600) // 60, s % 60
                                return f"{h:02d}:{m:02d}:{s2:02d}" if h else f"{m:02d}:{s2:02d}"
                            ts = f" @ {_fmt(c['start_ts'])} – {_fmt(c['end_ts'])}"
                        with st.container(border=True):
                            st.caption(f"📌 **{title}** — *{section}*{ts}")
                            st.write(c.get("snippet", ""))
                else:
                    st.warning("⚠️ Not found in the knowledge base.")
                    st.caption(result.get("answer", ""))
            except Exception as exc:
                st.error(f"Query error: {exc}")

    st.markdown("---")
    st.subheader("🔎 Debug Search (raw chunks)")
    debug_q = st.text_input("Search query (no LLM)")
    if st.button("Search") and debug_q:
        try:
            chunks = api_post("/search", json={"query": debug_q, "top_k": 5})
            for c in chunks:
                with st.expander(
                    f"[{c.get('rrf_score', 0):.4f}] {c.get('document_title', '')} — "
                    f"{c.get('section', '')}"
                ):
                    st.write(c.get("text", "")[:800])
        except Exception as e:
            st.error(str(e))
