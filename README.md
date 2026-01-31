# AskSEC 📊

A RAG-powered chatbot that answers questions about SEC 10-K filings using local AI.

[![CI](https://github.com/PhunsokNorboo/AskSEC/actions/workflows/ci.yml/badge.svg)](https://github.com/PhunsokNorboo/AskSEC/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## What It Does

Ask natural language questions about company filings:
- *"What are Apple's main risk factors?"*
- *"How does Netflix generate revenue?"*
- *"Compare Nvidia and Meta's AI strategies"*

The system retrieves relevant passages from SEC 10-K filings and generates accurate, cited answers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT WEB UI                            │
│                    (Chat Interface)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Query      │ -> │  Retrieval   │ -> │  Generation  │      │
│  │ Processing   │    │  (ChromaDB)  │    │  (Llama 3.2) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  SEC EDGAR   │ -> │   Parser &   │ -> │   ChromaDB   │      │
│  │  (edgartools)│    │   Chunker    │    │ (Embeddings) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Ollama + Llama 3.2 (local, free) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **Framework** | LangChain |
| **Data Source** | SEC EDGAR |
| **UI** | Streamlit |
| **Testing** | pytest + coverage |
| **CI/CD** | GitHub Actions |

**Total Cost: $0** - Everything runs locally.

## Companies Indexed

`AAPL` `MSFT` `GOOGL` `META` `AMZN` `NVDA` `TSLA` `NFLX` `V` `PYPL` `JNJ` `WMT`

12 companies × 2 filings each = 24 10-K reports

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed

### Installation

```bash
# Clone the repo
git clone https://github.com/PhunsokNorboo/AskSEC.git
cd AskSEC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Llama 3.2
ollama pull llama3.2

# Configure environment
cp .env.example .env
# Edit .env with your details
```

### Build the Database

```bash
# Download SEC filings
python scripts/download_filings.py

# Process documents
python scripts/process_filings.py

# Build vector database
python scripts/build_vectorstore.py
```

### Run the App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

## Docker

Run with Docker Compose (includes Ollama):

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

## Project Structure

```
AskSEC/
├── app/
│   └── streamlit_app.py      # Web interface
├── src/
│   ├── data/                 # Download & processing
│   │   ├── downloader.py     # SEC EDGAR downloader
│   │   ├── parser.py         # 10-K section extraction
│   │   └── preprocessor.py   # Document chunking
│   ├── embeddings/
│   │   └── vector_store.py   # ChromaDB operations
│   ├── rag/
│   │   ├── chain.py          # RAG pipeline
│   │   └── prompts.py        # Prompt templates
│   └── utils/
│       ├── config.py         # Configuration
│       └── logger.py         # Logging setup
├── tests/                    # Test suite
├── scripts/                  # Build scripts
├── data/
│   ├── raw/                  # SEC filings
│   ├── processed/            # Chunked documents
│   └── chroma_db/            # Vector database
├── .github/workflows/        # CI/CD
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_parser.py -v
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Type checking
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Features

- **Chat Interface** - Natural conversation with follow-ups
- **Company Filter** - Focus on specific companies
- **Source Citations** - See which filings were used
- **Example Questions** - One-click query examples
- **100% Local** - No API costs, data stays private

## Sample Output

**Q: What are Tesla's key risk factors?**

> Based on Tesla's 2025 10-K filing, key risk factors include:
> - Delays in launching and ramping production
> - Competition from other EV manufacturers
> - Limited charging infrastructure
> - Volatility in energy costs
> - Regulatory changes affecting EV incentives
>
> *Sources: TSLA 10-K 2025-01-30, Item 1A Risk Factors*

## Troubleshooting

### Ollama not running
```bash
# Start Ollama service
ollama serve

# Verify model is available
ollama list
```

### Empty responses
- Ensure vector database is built: `python scripts/build_vectorstore.py`
- Check that Ollama is running on port 11434

### Slow responses
- First query loads embedding model (takes ~10s)
- Subsequent queries are faster
- Consider using GPU acceleration for Ollama

## Future Improvements

- [ ] Add 10-Q quarterly reports
- [ ] Financial table extraction
- [ ] Multi-year trend analysis
- [ ] Deploy to cloud
- [ ] Add conversation memory
- [ ] Implement semantic caching

## License

MIT

## Author

Built by [Phunsok Norboo](https://github.com/PhunsokNorboo)
