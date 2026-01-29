"""RAG Chain - Connects retrieval with Ollama LLM generation."""
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

from src.embeddings.vector_store import VectorStoreManager
from src.rag.prompts import RAG_PROMPT, CHAT_RAG_PROMPT, format_documents
from src.utils.config import settings


class SECFilingRAG:
    """RAG pipeline for SEC filing analysis using Ollama."""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.1,
        vector_store_path: str = None
    ):
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
        self.chat_history = []

        print("RAG pipeline ready!")

    def _build_chain(self):
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
        filter_ticker: Optional[str] = None,
        k: int = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User's question
            filter_ticker: Optional ticker to filter results (e.g., "AAPL")
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer and source documents
        """
        k = k or settings.RETRIEVAL_K

        # Build search kwargs
        search_kwargs = {"k": k}
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

        # Generate answer
        prompt = CHAT_RAG_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = response.content

        # Format sources
        sources = self._format_sources(docs)

        # Update chat history
        self.chat_history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(docs)
        }

    def _format_sources(self, documents: List[Document]) -> List[Dict]:
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

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []

    def get_available_companies(self) -> List[str]:
        """Get list of companies in the database."""
        return self.vector_store_manager.get_available_companies()

    def search_only(
        self,
        query: str,
        k: int = 5,
        filter_ticker: Optional[str] = None
    ) -> List[Dict]:
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
