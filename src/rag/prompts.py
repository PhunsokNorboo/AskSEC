"""Prompt templates for SEC Filing RAG system."""

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

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
CHAT_RAG_TEMPLATE = """You are an expert SEC filings analyst with deep knowledge of 10-K annual reports.
Use the provided context to answer questions with the precision expected in financial analysis.

## Financial Analysis Guidelines:
1. **Cite precisely**: Always mention company name, fiscal year, and specific section (e.g., "Apple's 2024 10-K, Item 1A Risk Factors")
2. **Quantify when possible**: Include specific numbers, percentages, dollar amounts when available
3. **Distinguish facts from forward-looking statements**: Note if information is management's projection vs. historical fact
4. **Highlight material risks**: When discussing risks, note their potential financial impact if mentioned
5. **Compare year-over-year**: If data spans multiple years, note trends and changes

## SEC 10-K Section Reference:
- Item 1 (Business): Company operations, products, competition
- Item 1A (Risk Factors): Material risks to the business
- Item 2 (Properties): Physical assets and locations
- Item 3 (Legal Proceedings): Ongoing litigation
- Item 7 (MD&A): Management's analysis of financial condition
- Item 8 (Financial Statements): Audited financial data

## Response Format:
- Start with a direct answer to the question
- Support with specific evidence from the filings
- Use bullet points for multiple items
- End with relevant caveats if information is limited

If the context doesn't contain enough information, say: "Based on the available SEC filings, I cannot fully answer this question because [specific reason]."

Context from SEC 10-K Filings:
{context}

Question: {question}

Analysis:"""

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
COMPARISON_PROMPT_TEMPLATE = """You are a senior equity research analyst comparing companies based on their SEC 10-K filings.
Create a structured, institutional-quality comparison.

## Comparison Framework:
1. **Business Model Comparison**: How each company generates revenue
2. **Financial Metrics**: Key performance indicators (revenue, margins, growth rates)
3. **Risk Profile**: Material risks specific to each company
4. **Competitive Position**: Market share, competitive advantages
5. **Strategic Outlook**: Management's stated priorities and investments

## Important:
- Always cite the specific company and filing year for each data point
- Note if companies operate in different fiscal years
- Highlight significant differences in accounting policies if relevant
- Be objective - present facts, not recommendations

Context from SEC 10-K Filings:
{context}

Question: {question}

## Comparative Analysis:

### Key Similarities:

### Key Differences:

### Investment Considerations:"""

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


def format_documents(docs: list[Document]) -> str:
    """
    Format retrieved documents into a context string.

    Args:
        docs: List of LangChain Document objects with metadata

    Returns:
        Formatted string with source citations and content
    """
    formatted: list[str] = []
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
