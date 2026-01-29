# SECInsight - Project Status Report

## Project Overview

**SECInsight** is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about SEC 10-K filings from major public companies. Built as a portfolio project demonstrating skills in AI/ML engineering, data engineering, and financial domain knowledge.

**Status:** ✅ Complete (All 6 Phases)

**Total Cost:** $0 (100% free, local stack)

---

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **LLM** | Ollama + Llama 3.2 | Local, free, 2GB model |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Local, 384 dimensions |
| **Vector Database** | ChromaDB | Local storage, ~49MB |
| **RAG Framework** | LangChain | LCEL chains |
| **Data Source** | SEC EDGAR (via edgartools) | Free API |
| **Web UI** | Streamlit | Interactive chat interface |
| **Language** | Python 3.11 | Virtual environment |

---

## Project Structure

```
sec-filing-rag/
├── app/
│   └── streamlit_app.py      # Main Streamlit web application
├── data/
│   ├── raw/                   # Downloaded SEC filings (24 files)
│   ├── processed/             # Chunked documents (all_documents.pkl)
│   └── chroma_db/             # Vector database (~49MB)
├── scripts/
│   ├── download_filings.py    # Download SEC 10-K filings
│   ├── process_filings.py     # Parse and chunk documents
│   ├── build_vectorstore.py   # Build ChromaDB with embeddings
│   └── test_rag.py            # Test RAG pipeline
├── src/
│   ├── data/
│   │   ├── downloader.py      # SEC filing download logic
│   │   ├── parser.py          # 10-K section extraction
│   │   └── preprocessor.py    # Text chunking
│   ├── embeddings/
│   │   └── vector_store.py    # ChromaDB operations
│   ├── rag/
│   │   ├── chain.py           # RAG pipeline with Ollama
│   │   └── prompts.py         # Prompt templates
│   └── utils/
│       └── config.py          # Configuration management
├── .env                        # Environment variables (not committed)
├── .env.example               # Template for .env
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── run_app.sh                 # Convenience script to run app
└── CLAUDE.md                  # This file
```

---

## Data Summary

### Companies Indexed (12)

| Company | Ticker | Sector | Filings |
|---------|--------|--------|---------|
| Apple | AAPL | Tech | 2 |
| Microsoft | MSFT | Tech | 2 |
| Alphabet (Google) | GOOGL | Tech | 2 |
| Meta | META | Social Media | 2 |
| Amazon | AMZN | E-commerce/Cloud | 2 |
| Nvidia | NVDA | Semiconductors | 2 |
| Tesla | TSLA | Automotive/Tech | 2 |
| Netflix | NFLX | Entertainment | 2 |
| Visa | V | Fintech/Payments | 2 |
| PayPal | PYPL | Fintech/Digital | 2 |
| Johnson & Johnson | JNJ | Healthcare | 2 |
| Walmart | WMT | Retail | 2 |

### Processing Stats

| Metric | Value |
|--------|-------|
| Total filings | 24 |
| Total chunks | 6,196 (in vector DB) |
| Avg chunk size | ~714 characters |
| Raw data size | ~10 MB |
| Vector DB size | ~49 MB |

---

## Implementation Phases

### Phase 1: Project Setup ✅
- Created project directory structure
- Set up Python virtual environment
- Installed dependencies (free stack)
- Configured environment variables
- Installed Ollama + Llama 3.2

### Phase 2: Data Acquisition ✅
- Built SEC filing downloader using edgartools
- Downloaded 24 10-K filings from 12 companies
- Saved raw text + metadata JSON for each filing
- Replaced JPMorgan (empty files) with Visa

### Phase 3: Document Processing ✅
- Created 10-K section parser (extracts Items 1, 1A, 7, 7A, 8, etc.)
- Built document chunker with RecursiveCharacterTextSplitter
- Processed all filings into 11,564 chunks
- Saved processed documents as pickle file

### Phase 4: Vector Database ✅
- Set up ChromaDB with persistent storage
- Integrated sentence-transformers for local embeddings
- Built vector store from processed documents
- Tested similarity search functionality

### Phase 5: RAG Pipeline ✅
- Created prompt templates for financial analysis
- Built RAG chain connecting retrieval + Llama 3.2
- Implemented company filtering
- Added source citation formatting
- Tested with multiple query types

### Phase 6: Streamlit Application ✅
- Built interactive web UI with chat interface
- Added company filter dropdown
- Implemented example questions
- Added source citations with expandable details
- Created run script for easy startup

---

## How to Run

### Prerequisites
1. Python 3.11+
2. Ollama installed with Llama 3.2 (`ollama pull llama3.2`)

### Quick Start
```bash
# Navigate to project
cd /Users/phunsoknorboo/RAG/sec-filing-rag

# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app/streamlit_app.py

# Open browser to http://localhost:8501
```

### Alternative: Use run script
```bash
./run_app.sh
```

### Rebuild from Scratch
```bash
# 1. Download filings
python scripts/download_filings.py

# 2. Process documents
python scripts/process_filings.py

# 3. Build vector database
python scripts/build_vectorstore.py

# 4. Test RAG pipeline
python scripts/test_rag.py

# 5. Run web app
streamlit run app/streamlit_app.py
```

---

## Configuration

### Environment Variables (.env)
```
EDGAR_IDENTITY="Your Name your.email@example.com"
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Key Settings (src/utils/config.py)
- `CHUNK_SIZE`: 1000 characters
- `CHUNK_OVERLAP`: 200 characters
- `RETRIEVAL_K`: 6 documents

---

## Sample Queries

The RAG system can answer questions like:
- "What are Apple's main business segments?"
- "What are Tesla's key risk factors?"
- "How does Netflix generate revenue?"
- "What does Nvidia say about AI demand?"
- "Compare Meta and Google's advertising business"
- "What cybersecurity risks does PayPal face?"

---

## Known Limitations

1. **Section Extraction**: Parser primarily extracts Item 8 (Financial Statements) from most filings due to varied formatting in 10-K documents
2. **Context Window**: Limited by chunk retrieval (6 chunks default)
3. **Cross-Company Comparisons**: Work best when both companies have relevant content in retrieved chunks
4. **Response Time**: Local LLM inference takes 5-15 seconds per query

---

## Future Improvements

- [ ] Add 10-Q quarterly reports
- [ ] Improve section parser regex patterns
- [ ] Add financial table extraction
- [ ] Implement multi-year trend analysis
- [ ] Add earnings call transcripts
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Add conversation memory for follow-up questions

---

## File Locations

| What | Path |
|------|------|
| Project root | `/Users/phunsoknorboo/RAG/sec-filing-rag/` |
| Virtual environment | `./venv/` |
| Raw SEC filings | `./data/raw/` |
| Processed documents | `./data/processed/all_documents.pkl` |
| Vector database | `./data/chroma_db/` |
| Streamlit app | `./app/streamlit_app.py` |

---

## Created By

Built with Claude Code as a portfolio project for job applications.

**Date Completed:** January 29, 2026
