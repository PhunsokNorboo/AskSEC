# AskSEC 📊

A RAG-powered chatbot that answers questions about SEC 10-K filings using local AI.

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

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Ollama + Llama 3.2 (local, free) |
| **Embeddings** | sentence-transformers |
| **Vector DB** | ChromaDB |
| **Framework** | LangChain |
| **Data Source** | SEC EDGAR |
| **UI** | Streamlit |

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

## Project Structure

```
AskSEC/
├── app/
│   └── streamlit_app.py      # Web interface
├── src/
│   ├── data/                 # Download & processing
│   ├── embeddings/           # Vector store operations
│   └── rag/                  # RAG chain & prompts
├── scripts/                  # Build scripts
├── data/
│   ├── raw/                  # SEC filings
│   └── chroma_db/            # Vector database
└── requirements.txt
```

## How It Works

```
User Question
     │
     ▼
┌─────────────┐
│  Retriever  │ ◄── ChromaDB (semantic search)
└─────────────┘
     │
     ▼
┌─────────────┐
│   Prompt    │ ◄── Context + Question
└─────────────┘
     │
     ▼
┌─────────────┐
│   Llama 3.2 │ ◄── Local LLM via Ollama
└─────────────┘
     │
     ▼
Answer + Sources
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

## Future Improvements

- [ ] Add 10-Q quarterly reports
- [ ] Financial table extraction
- [ ] Multi-year trend analysis
- [ ] Deploy to cloud

## License

MIT

## Author

Built by [Phunsok Norboo](https://github.com/PhunsokNorboo)
