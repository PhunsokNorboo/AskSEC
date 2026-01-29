"""SEC Filing RAG Chatbot - Streamlit Application."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.rag.chain import SECFilingRAG

# Page configuration
st.set_page_config(
    page_title="SEC Filing Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }

    /* Chat messages */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    /* Source cards */
    .source-card {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        padding: 1rem;
    }

    /* Example question buttons */
    .stButton > button {
        width: 100%;
        text-align: left;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached to avoid reloading)."""
    return SECFilingRAG()


def display_sources(sources):
    """Display source documents in an expandable section."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            **{i}. {source['company']}** (`{source['ticker']}`)
            - 📅 Filing Date: {source['filing_date']}
            - 📄 Section: {source['section']}
            - 📝 Excerpt: _{source['excerpt'][:150]}..._
            """)
            st.divider()


def main():
    """Main application."""

    # Initialize RAG system
    try:
        rag = initialize_rag()
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.info("Make sure Ollama is running with `ollama serve` and the vector database is built.")
        return

    # Get available companies
    available_companies = rag.get_available_companies()

    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("⚙️ Settings")

        # Company filter
        filter_options = ["All Companies"] + available_companies
        selected_company = st.selectbox(
            "🏢 Filter by Company",
            options=filter_options,
            index=0,
            help="Filter responses to a specific company's filings"
        )

        # Number of sources
        num_sources = st.slider(
            "📑 Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=6,
            help="More sources = more context but slower responses"
        )

        st.divider()

        # Example questions
        st.subheader("💡 Example Questions")

        example_questions = [
            "What are Apple's main business segments?",
            "What are Tesla's key risk factors?",
            "How does Netflix generate revenue?",
            "What does Nvidia say about AI demand?",
            "Compare Meta and Google's advertising business",
            "What cybersecurity risks does PayPal face?",
            "How has Walmart's e-commerce grown?",
            "What is Microsoft's cloud strategy?",
        ]

        for question in example_questions:
            if st.button(question, key=f"ex_{hash(question)}", use_container_width=True):
                st.session_state.pending_question = question

        st.divider()

        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            rag.clear_history()
            st.rerun()

        # About section
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        This app uses **RAG** (Retrieval-Augmented Generation) to answer
        questions about SEC 10-K filings.

        **Tech Stack:**
        - 🦙 Llama 3.2 (via Ollama)
        - 🔍 ChromaDB vector search
        - 📊 12 companies indexed
        - 💰 100% free & local
        """)

    # ===== MAIN CONTENT =====

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 SEC Filing Analyzer</h1>
        <p><em>Ask questions about company 10-K filings using AI</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Company badges
    st.markdown("**Companies indexed:** " + " • ".join([f"`{c}`" for c in available_companies]))
    st.divider()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check for pending question from sidebar
    pending_question = st.session_state.pop("pending_question", None)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])

    # Chat input
    user_input = st.chat_input("Ask a question about SEC filings...") or pending_question

    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching filings and generating answer..."):
                # Determine filter
                filter_ticker = None if selected_company == "All Companies" else selected_company

                # Query RAG
                try:
                    result = rag.query(
                        question=user_input,
                        filter_ticker=filter_ticker,
                        k=num_sources
                    )

                    # Display answer
                    st.markdown(result["answer"])

                    # Display sources
                    display_sources(result["sources"])

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

    # Footer
    st.markdown("""
    <div class="footer">
        Built with Streamlit, LangChain, ChromaDB, and Ollama<br>
        Data source: SEC EDGAR 10-K Filings
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
