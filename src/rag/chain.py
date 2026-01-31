"""RAG Chain - Connects retrieval with Ollama LLM generation."""
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from src.embeddings.vector_store import VectorStoreManager
from src.rag.prompts import CHAT_RAG_PROMPT, format_documents
from src.utils.config import settings

logger = logging.getLogger("asksec.rag")

# Valid ticker pattern (1-5 uppercase letters)
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")
MAX_QUESTION_LENGTH = 2000


class SECFilingRAG:
    """RAG pipeline for SEC filing analysis using Ollama."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        vector_store_path: str | None = None
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            model_name: Ollama model to use (default from config)
            temperature: LLM temperature (lower = more focused)
            vector_store_path: Path to ChromaDB storage
        """
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.temperature = temperature

        # Initialize LLM (Ollama with Llama 3.2)
        print(f"Initializing LLM: {self.model_name}")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=temperature,
        )

        # Initialize vector store
        print("Loading vector store...")
        self.vector_store_manager = VectorStoreManager(
            persist_directory=vector_store_path or str(settings.CHROMA_DB_DIR)
        )
        self.vector_store_manager.create_or_load_store()

        # Build the chain
        self.chain = self._build_chain()
        self.chat_history: list[dict[str, str]] = []

        print("RAG pipeline ready!")

    def _validate_ticker(self, ticker: str | None) -> str | None:
        """
        Validate and normalize a ticker symbol.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            Normalized ticker or None if invalid

        Raises:
            ValueError: If ticker format is invalid
        """
        if ticker is None:
            return None
        ticker = ticker.strip().upper()
        if not TICKER_PATTERN.match(ticker):
            raise ValueError(f"Invalid ticker format: {ticker}. Expected 1-5 uppercase letters.")
        return ticker

    def _validate_question(self, question: str) -> str:
        """
        Validate and sanitize user question.

        Args:
            question: User's question

        Returns:
            Sanitized question

        Raises:
            ValueError: If question is empty or too long
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        question = question.strip()
        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed.")
        return question

    def _build_chain(self) -> Any:
        """Build the RAG chain using LCEL (LangChain Expression Language)."""
        retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": settings.RETRIEVAL_K}
        )

        # RAG chain: retrieve -> format -> prompt -> llm -> parse
        chain = (
            RunnableParallel(
                context=retriever | format_documents,
                question=RunnablePassthrough()
            )
            | CHAT_RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(
        self,
        question: str,
        filter_ticker: str | None = None,
        k: int | None = None
    ) -> dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User's question
            filter_ticker: Optional ticker to filter results (e.g., "AAPL")
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer and source documents

        Raises:
            ValueError: If question or ticker is invalid
        """
        # Validate inputs
        question = self._validate_question(question)
        filter_ticker = self._validate_ticker(filter_ticker)

        k = k or settings.RETRIEVAL_K

        # Build search kwargs
        search_kwargs: dict[str, Any] = {"k": k}
        if filter_ticker:
            search_kwargs["filter"] = {"ticker": filter_ticker}

        # Get retriever with filters
        retriever = self.vector_store_manager.get_retriever(
            search_kwargs=search_kwargs
        )

        # Retrieve relevant documents
        docs = retriever.invoke(question)

        # Format context
        context = format_documents(docs)

        # Generate answer with error handling
        try:
            prompt = CHAT_RAG_PROMPT.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            answer = "I encountered an error generating a response. Please try again."

        # Format sources
        sources = self._format_sources(docs)

        # Update chat history
        self.chat_history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(docs)
        }

    def _format_sources(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Format source documents for display."""
        sources = []
        seen = set()

        for doc in documents:
            source_key = (
                doc.metadata.get("ticker"),
                doc.metadata.get("filing_date"),
                doc.metadata.get("item_number")
            )

            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "ticker": doc.metadata.get("ticker"),
                    "company": doc.metadata.get("company_name"),
                    "filing_date": doc.metadata.get("filing_date"),
                    "section": doc.metadata.get("item_title"),
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })

        return sources

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history = []

    def get_available_companies(self) -> list[str]:
        """Get list of companies in the database."""
        return self.vector_store_manager.get_available_companies()

    def search_only(
        self,
        query: str,
        k: int = 5,
        filter_ticker: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Search without generating an answer (useful for debugging).

        Args:
            query: Search query
            k: Number of results
            filter_ticker: Optional ticker filter

        Returns:
            List of search results with metadata
        """
        search_kwargs = {"k": k}
        if filter_ticker:
            search_kwargs["filter"] = {"ticker": filter_ticker}

        docs = self.vector_store_manager.similarity_search(
            query=query,
            k=k,
            filter_dict={"ticker": filter_ticker} if filter_ticker else None
        )

        return [
            {
                "content": doc.page_content,
                "ticker": doc.metadata.get("ticker"),
                "company": doc.metadata.get("company_name"),
                "filing_date": doc.metadata.get("filing_date"),
                "section": doc.metadata.get("item_title"),
            }
            for doc in docs
        ]
