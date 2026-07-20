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

    header = f"**{index}. {filename}**"
    if page not in (None, "", "N/A", 0):
        header += f" — Page {page}"

    snippet = text[:200] + "..." if len(text) > 200 else text

    return f"{header}\n\n> {snippet}"


# --- Sidebar ---


def render_sidebar():
    """Render the sidebar with agent info and status."""
    with st.sidebar:
        st.title(AGENT_NAME)
        st.markdown(AGENT_DESCRIPTION)

        st.divider()

        # Status indicators
        st.subheader("Status")
        rag = st.session_state.get("rag")
        error = st.session_state.get("rag_error")

        if error:
            st.error(f"Connection error: {error}")
        elif rag:
            doc_count = get_document_count(rag)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", doc_count)
            with col2:
                st.metric("Collection", MONORAG_COLLECTION)

            st.caption(f"Model: `{LLM_MODEL}`")
            st.caption(f"Provider: `{LLM_PROVIDER}`")

            if doc_count == 0:
                st.warning(
                    "No documents indexed yet. Use the MonoRAG CLI to index documents."
                )

        st.divider()

        # Suggested questions
        st.subheader("Suggested questions")
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, key=f"suggest_{question[:20]}", use_container_width=True):
                st.session_state["pending_question"] = question
                st.rerun()

        st.divider()
        st.caption("Powered by [MonoRAG](https://github.com/BryanInfante/monorag)")


# --- Chat Interface ---


def render_chat():
    """Render the main chat interface."""
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources", expanded=False):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(format_citation(src, i))

    # Handle pending question from suggested questions
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = st.chat_input("Ask a question about the indexed documents...")
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
                error = st.session_state.get("rag_error", "Unknown connection error")
                response = f"I cannot process your question right now. Error: {error}"
                st.markdown(response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )
            else:
                with st.spinner("Thinking..."):
                    try:
                        result = rag.ask(query, top_k=5)
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)
                        if sources:
                            with st.expander("Sources", expanded=False):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(format_citation(src, i))

                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    except RuntimeError as e:
                        error_msg = (
                            f"An error occurred while generating the response: {e}\n\n"
                            "Please try again in a moment."
                        )
                        st.error(error_msg)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": error_msg}
                        )


# --- Main ---


def main():
    """Application entry point."""
    get_rag_module()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
