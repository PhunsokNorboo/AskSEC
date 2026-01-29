"""Prompt templates for SEC Filing RAG system."""
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Main RAG prompt for answering questions
RAG_PROMPT_TEMPLATE = """You are a financial analyst assistant specializing in SEC filings analysis.
Use the following pieces of context from SEC 10-K filings to answer the question.

Important guidelines:
1. Only use information from the provided context
2. If you cannot find the answer in the context, say so clearly
3. Cite the source (company name, filing date, section) when making claims
4. Be specific and provide quantitative data when available
5. If comparing companies, structure your response clearly

Context from SEC 10-K Filings:
{context}

Question: {question}

Helpful Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# Chat-style RAG prompt (for conversational interface)
CHAT_RAG_TEMPLATE = """You are a financial analyst assistant specializing in SEC filings analysis.
Use the provided context from SEC 10-K filings to answer questions accurately.

Guidelines:
- Only use information from the provided context
- If the answer isn't in the context, say "I don't have enough information to answer that"
- Cite sources: mention company name and filing date
- Be concise but thorough
- Use bullet points for clarity when listing multiple items

Context:
{context}

Question: {question}

Answer:"""

CHAT_RAG_PROMPT = PromptTemplate(
    template=CHAT_RAG_TEMPLATE,
    input_variables=["context", "question"]
)


# Prompt for condensing follow-up questions (for conversation history)
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question that captures all relevant context.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    template=CONDENSE_QUESTION_TEMPLATE,
    input_variables=["chat_history", "question"]
)


# Prompt for comparing multiple companies
COMPARISON_PROMPT_TEMPLATE = """You are a financial analyst comparing SEC 10-K filings from multiple companies.
Analyze the provided context and create a structured comparison.

Context from SEC 10-K Filings:
{context}

Question: {question}

Provide a structured comparison with:
1. Key similarities
2. Key differences
3. Summary insights

Comparison:"""

COMPARISON_PROMPT = PromptTemplate(
    template=COMPARISON_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# Prompt for summarizing a specific section
SUMMARY_PROMPT_TEMPLATE = """You are a financial analyst summarizing SEC 10-K filing content.
Create a concise summary of the following information.

Context:
{context}

Topic: {question}

Provide a clear, structured summary with key points:"""

SUMMARY_PROMPT = PromptTemplate(
    template=SUMMARY_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def format_documents(docs) -> str:
    """Format retrieved documents into a context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        ticker = doc.metadata.get('ticker', 'Unknown')
        company = doc.metadata.get('company_name', 'Unknown')
        date = doc.metadata.get('filing_date', 'Unknown')
        section = doc.metadata.get('item_title', 'Unknown')

        formatted.append(
            f"[Source {i}: {company} ({ticker}), Filed: {date}, Section: {section}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)
