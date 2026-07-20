"""Inspector END — Streamlit chat interface for MonoRAG.

A read-only chat UI that connects to a pre-indexed MonoRAG collection
and answers questions using the Inspector END personality.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import (
    AGENT_DESCRIPTION,
    AGENT_NAME,
    AGENT_SUBTITLE,
    DISCLAIMER,
    EMPTY_STATE_DESCRIPTION,
    EMPTY_STATE_TITLE,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    MONORAG_COLLECTION,
    MONORAG_DB_PATH,
    SUGGESTED_QUESTIONS,
)
from agent_generator import InspectorGenerator

# --- Page Configuration ---

st.set_page_config(
    page_title=AGENT_NAME,
    page_icon="🔍",
    layout="centered",
)

# --- Custom CSS ---

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ie-primary: #4F46E5;
    --ie-primary-dark: #4338CA;
    --ie-primary-tint: #eef2ff;
    --ie-primary-border: #e0e4ff;
    --ie-accent: #0d9488;
    --ie-text: #111827;
    --ie-muted: #6b7280;
    --ie-faint: #9ca3af;
}

.stApp {
    font-family: 'Source Sans 3', -apple-system, system-ui, sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #f5f6f9;
    border-right: 1px solid #e6e8ee;
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.01em;
}

/* Status connected badge */
.status-connected {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    font-weight: 600;
    color: #0d9488;
    letter-spacing: 0.02em;
}
.status-connected::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #0d9488;
    box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.16);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e6e8ee;
    border-radius: 10px;
    padding: 10px 12px;
}
div[data-testid="stMetric"] label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #6b7280;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #0d9488;
    font-weight: 700;
}

/* Suggested question buttons */
.stButton > button {
    text-align: left;
    font-size: 13px;
    line-height: 1.35;
    color: #312e81;
    background: #eef2ff;
    border: 1px solid #e0e4ff;
    border-radius: 9px;
    padding: 9px 12px;
    transition: background 0.15s, border-color 0.15s;
}
.stButton > button:hover {
    background: #e4e9ff;
    border-color: #c9d0ff;
    color: #312e81;
}

/* Chat message styling */
.stChatMessage[data-testid="stChatMessage"] {
    border-radius: 14px;
    font-size: 15px;
    line-height: 1.6;
}

/* Sources expander */
div[data-testid="stExpander"] {
    border: 1px solid #e6e8ee;
    border-radius: 9px;
    background: #fbfbfd;
}
div[data-testid="stExpander"] summary {
    font-size: 13px;
    font-weight: 600;
    color: #4F46E5;
}

/* Source citation blocks */
.source-citation {
    border: 1px solid #e6e8ee;
    border-radius: 9px;
    padding: 11px 13px;
    background: #ffffff;
    margin-bottom: 8px;
}
.source-citation .source-file {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    font-weight: 500;
    color: #312e81;
}
.source-citation .source-page {
    font-size: 11px;
    color: #6b7280;
    background: #f0f2f6;
    border-radius: 6px;
    padding: 2px 7px;
    display: inline-block;
}
.source-citation blockquote {
    margin: 6px 0 2px 0;
    padding: 6px 0 2px 12px;
    border-left: 3px solid #c7d2fe;
    font-size: 13px;
    line-height: 1.5;
    color: #4b5563;
    font-style: italic;
}

/* Disclaimer */
.disclaimer {
    text-align: center;
    font-size: 11.5px;
    color: #b0b4bf;
    margin-top: 8px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 60px 28px;
}
.empty-state h2 {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    color: #111827;
}
.empty-state p {
    font-size: 16px;
    color: #6b7280;
    max-width: 460px;
    margin: 12px auto 30px;
    line-height: 1.55;
}

/* Config captions in sidebar */
.config-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: #8b90a0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Helper Functions ---


def get_rag_module():
    """Create or retrieve the cached RAGModule instance."""
    if "rag" not in st.session_state:
        if not LLM_API_KEY:
            st.session_state["rag"] = None
            st.session_state["rag_error"] = (
                "LLM_API_KEY is not configured. "
                "Set it in the .env file or as an environment variable."
            )
            return None

        try:
            from rag_core import RAGModule

            generator = InspectorGenerator(
                api_key=LLM_API_KEY,
                model=LLM_MODEL,
                provider_name=LLM_PROVIDER,
                base_url=LLM_BASE_URL,
            )

            kwargs = {
                "collection": MONORAG_COLLECTION,
                "max_history": 10,
                "generator": generator,
            }
            if MONORAG_DB_PATH:
                kwargs["db_path"] = MONORAG_DB_PATH

            rag = RAGModule(**kwargs)
            st.session_state["rag"] = rag
            st.session_state["rag_error"] = None
        except Exception as e:
            st.session_state["rag"] = None
            st.session_state["rag_error"] = str(e)

    return st.session_state.get("rag")


def get_document_count(rag) -> int:
    """Get the number of documents in the active collection."""
    try:
        return rag.retriever._collection.count()
    except Exception:
        return 0


def format_citation(source: dict, index: int) -> str:
    """Format a source citation with filename, page, and snippet."""
    metadata = source.get("metadata", {})
    text = source.get("text", "")

    filename = metadata.get("source", "Unknown")
    page = metadata.get("page")

    page_badge = ""
    if page not in (None, "", "N/A", 0):
        page_badge = f' <span class="source-page">p. {page}</span>'

    snippet = text[:200] + "..." if len(text) > 200 else text

    return (
        f'<div class="source-citation">'
        f'<div class="source-file">{filename}{page_badge}</div>'
        f"<blockquote>{snippet}</blockquote>"
        f"</div>"
    )


# --- Sidebar ---


def render_sidebar():
    """Render the sidebar with agent info and status."""
    with st.sidebar:
        st.markdown("# 🔍 Inspector END")
        st.markdown(f"*{AGENT_SUBTITLE}*")

        st.divider()

        # Status
        rag = st.session_state.get("rag")
        error = st.session_state.get("rag_error")

        if error:
            st.error(f"Error de conexión: {error}")
        elif rag:
            st.markdown('<div class="status-connected">CONECTADO</div>', unsafe_allow_html=True)
            st.write("")

            doc_count = get_document_count(rag)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documentos", doc_count)
            with col2:
                st.metric("Colección", MONORAG_COLLECTION)

            st.markdown(
                f'<div class="config-caption">'
                f"<b>Model:</b> {LLM_MODEL}<br>"
                f"<b>Provider:</b> {LLM_PROVIDER}"
                f"</div>",
                unsafe_allow_html=True,
            )

            if doc_count == 0:
                st.warning(
                    "No hay documentos indexados. "
                    "Usa el CLI de MonoRAG para indexar documentos."
                )

        st.divider()

        # Suggested questions
        st.markdown(
            '<div style="font-size:12px;font-weight:700;color:#374151;'
            'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">'
            "Preguntas sugeridas</div>",
            unsafe_allow_html=True,
        )
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, key=f"suggest_{hash(question)}", use_container_width=True):
                st.session_state["pending_question"] = question
                st.rerun()

        st.divider()
        st.markdown(
            '<div style="font-size:12px;color:#9ca3af;">'
            'Desarrollado con <a href="https://github.com/BryanInfante/monorag" '
            'style="color:#4F46E5;font-weight:600;text-decoration:none;">MonoRAG</a></div>',
            unsafe_allow_html=True,
        )


# --- Chat Interface ---


def render_chat():
    """Render the main chat interface."""
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Show empty state if no messages yet
    if not st.session_state["messages"]:
        st.markdown(
            f'<div class="empty-state">'
            f"<h2>{EMPTY_STATE_TITLE}</h2>"
            f"<p>{EMPTY_STATE_DESCRIPTION}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Fuentes ({len(message['sources'])})", expanded=False):
                    citations_html = "".join(
                        format_citation(src, i)
                        for i, src in enumerate(message["sources"], 1)
                    )
                    st.markdown(citations_html, unsafe_allow_html=True)

    # Handle pending question from suggested questions
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = st.chat_input("Haz una pregunta sobre los documentos indexados…")
    query = pending or user_input

    if query:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        rag = st.session_state.get("rag")
        with st.chat_message("assistant"):
            if rag is None:
                error = st.session_state.get("rag_error", "Error de conexión desconocido")
                response = f"No puedo procesar tu pregunta en este momento. Error: {error}"
                st.markdown(response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )
            else:
                with st.spinner("Pensando…"):
                    try:
                        result = rag.ask(query, top_k=5)
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)
                        if sources:
                            with st.expander(
                                f"Fuentes ({len(sources)})", expanded=False
                            ):
                                citations_html = "".join(
                                    format_citation(src, i)
                                    for i, src in enumerate(sources, 1)
                                )
                                st.markdown(citations_html, unsafe_allow_html=True)

                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    except RuntimeError as e:
                        error_msg = (
                            f"Ocurrió un error al generar la respuesta: {e}\n\n"
                            "Por favor intenta de nuevo en un momento."
                        )
                        st.error(error_msg)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": error_msg}
                        )

    # Disclaimer
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


# --- Main ---


def main():
    """Application entry point."""
    get_rag_module()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
