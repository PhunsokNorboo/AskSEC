#!/bin/bash
# Run the SEC Filing RAG Streamlit Application

# Change to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Run Streamlit
echo "🚀 Starting SEC Filing Analyzer..."
echo "   Open your browser to: http://localhost:8501"
echo ""
streamlit run app/streamlit_app.py --server.port 8501
