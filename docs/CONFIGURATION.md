# AskSEC Configuration Guide

This document describes all configuration options for the AskSEC application.

## Environment Variables

Configuration is managed through environment variables, loaded from a `.env` file.

### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `EDGAR_IDENTITY` | Your identity for SEC EDGAR API (required by SEC) | `"John Doe john@example.com"` |

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name to use for generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |

### RAG Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Maximum characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_K` | `6` | Number of documents to retrieve per query |

### ChromaDB Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_COLLECTION_NAME` | `sec_filings` | Name of the vector database collection |

## Configuration File

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
nano .env
```

### Example `.env` file

```env
# Required: Your identity for SEC EDGAR
EDGAR_IDENTITY="Your Name your.email@example.com"

# Optional: Customize LLM
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Customize embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional: Customize RAG parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=6

# Optional: Customize ChromaDB
CHROMA_COLLECTION_NAME=sec_filings
```

## Directory Structure

The application uses the following directories (auto-created):

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Downloaded SEC filings |
| `data/processed/` | Processed document chunks |
| `data/chroma_db/` | Vector database storage |

## Tuning Parameters

### Chunk Size

- **Smaller chunks** (500-800): Better for specific questions, more precise retrieval
- **Larger chunks** (1000-1500): Better for broad questions, more context

### Chunk Overlap

- **Smaller overlap** (100): Less redundancy, faster processing
- **Larger overlap** (300): Better continuity, fewer broken sentences

### Retrieval K

- **Fewer documents** (3-4): Faster responses, less context
- **More documents** (8-10): Richer context, slower responses

## Embedding Models

Compatible sentence-transformers models:

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `all-MiniLM-L6-v2` | 80MB | Good | Fast |
| `all-mpnet-base-v2` | 420MB | Better | Medium |
| `multi-qa-mpnet-base-dot-v1` | 420MB | Best for QA | Medium |

## Ollama Models

Compatible Ollama models:

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `llama3.2` | 2GB | Good | Fast |
| `llama3.1` | 4.7GB | Better | Medium |
| `llama3.1:70b` | 40GB | Best | Slow |
| `mistral` | 4.1GB | Good | Fast |

## Validation

Verify your configuration:

```python
from src.utils.config import settings

# Check if valid
if settings.validate():
    print("Configuration OK")
else:
    print("Please configure EDGAR_IDENTITY in .env")
```

## Troubleshooting

### "EDGAR_IDENTITY not configured"

Set your identity in `.env`:
```
EDGAR_IDENTITY="Your Name your.email@example.com"
```

### "Connection refused" (Ollama)

Start Ollama:
```bash
ollama serve
```

### Slow embedding

Try a smaller model:
```
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Out of memory

Reduce chunk retrieval:
```
RETRIEVAL_K=4
```
