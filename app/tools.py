"""
LangChain tools used by the LangGraph agent.
Each tool is a focused, callable function that the LLM can invoke.

Tools:
  - rag_retrieval_tool       : Retrieve relevant document chunks from ChromaDB
  - financial_calculator_tool: Perform financial ratio calculations
  - stock_data_tool          : Fetch live stock price data via yfinance
  - document_summary_tool    : Summarise a retrieved context window
  - web_search_tool          : Placeholder for web-search integration
"""

from typing import Optional
import json

import yfinance as yf
from langchain.tools import tool
from loguru import logger

from app.vector_store import get_vector_store


# ---------------------------------------------------------------------------
# RAG Retrieval
# ---------------------------------------------------------------------------

@tool
def rag_retrieval_tool(query: str, k: int = 5) -> str:
    """
    Retrieve the most relevant passages from the financial document store
    using semantic similarity search.

    Args:
        query: The search question or topic.
        k: Number of document chunks to retrieve (default 5).

    Returns:
        A formatted string containing retrieved passages with source metadata.
    """
    try:
        vs = get_vector_store()
        results = vs.similarity_search_with_score(query, k=k)
        if not results:
            return "No relevant documents found in the knowledge base."

        passages = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            passages.append(
                f"[{i}] Source: {source} | Page: {page} | Score: {score:.3f}\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(passages)
    except Exception as e:
        logger.error("RAG retrieval failed: {}", e)
        return f"Error during retrieval: {str(e)}"


# ---------------------------------------------------------------------------
# Financial Calculator
# ---------------------------------------------------------------------------

@tool
def financial_calculator_tool(data: str) -> str:
    """
    Perform common financial ratio and metric calculations.

    Args:
        data: JSON string with 'metric' and required numeric fields.

    Supported metrics:
      - pe_ratio        : {"metric": "pe_ratio", "price": X, "eps": Y}
      - profit_margin   : {"metric": "profit_margin", "net_income": X, "revenue": Y}
      - debt_to_equity  : {"metric": "debt_to_equity", "total_debt": X, "equity": Y}
      - current_ratio   : {"metric": "current_ratio", "current_assets": X, "current_liabilities": Y}
      - roe             : {"metric": "roe", "net_income": X, "equity": Y}
      - cagr            : {"metric": "cagr", "start_value": X, "end_value": Y, "years": N}
      - eps             : {"metric": "eps", "net_income": X, "shares": Y}

    Returns:
        Computed result as a formatted string.
    """
    try:
        params = json.loads(data)
        metric = params.get("metric", "").lower()

        if metric == "pe_ratio":
            price, eps = float(params["price"]), float(params["eps"])
            if eps == 0:
                return "EPS is zero — P/E ratio undefined."
            result = price / eps
            return f"P/E Ratio = {result:.2f}"

        elif metric == "profit_margin":
            net_income = float(params["net_income"])
            revenue = float(params["revenue"])
            if revenue == 0:
                return "Revenue is zero — margin undefined."
            result = (net_income / revenue) * 100
            return f"Profit Margin = {result:.2f}%"

        elif metric == "debt_to_equity":
            debt = float(params["total_debt"])
            equity = float(params["equity"])
            if equity == 0:
                return "Equity is zero — D/E ratio undefined."
            result = debt / equity
            return f"Debt-to-Equity Ratio = {result:.2f}"

        elif metric == "current_ratio":
            ca = float(params["current_assets"])
            cl = float(params["current_liabilities"])
            if cl == 0:
                return "Current liabilities are zero — ratio undefined."
            result = ca / cl
            return f"Current Ratio = {result:.2f}"

        elif metric == "roe":
            net_income = float(params["net_income"])
            equity = float(params["equity"])
            if equity == 0:
                return "Equity is zero — ROE undefined."
            result = (net_income / equity) * 100
            return f"Return on Equity (ROE) = {result:.2f}%"

        elif metric == "cagr":
            sv = float(params["start_value"])
            ev = float(params["end_value"])
            years = float(params["years"])
            if sv <= 0 or years <= 0:
                return "Invalid inputs for CAGR."
            result = ((ev / sv) ** (1 / years) - 1) * 100
            return f"CAGR = {result:.2f}% over {years:.0f} years"

        elif metric == "eps":
            net_income = float(params["net_income"])
            shares = float(params["shares"])
            if shares == 0:
                return "Shares outstanding is zero — EPS undefined."
            result = net_income / shares
            return f"EPS = {result:.2f}"

        else:
            return (
                f"Unknown metric '{metric}'. "
                "Supported: pe_ratio, profit_margin, debt_to_equity, "
                "current_ratio, roe, cagr, eps."
            )
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return f"Calculation error: {str(e)}"


# ---------------------------------------------------------------------------
# Stock Data
# ---------------------------------------------------------------------------

@tool
def stock_data_tool(ticker: str) -> str:
    """
    Fetch real-time stock information and key financial metrics for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA').

    Returns:
        Formatted string with current price, market cap, P/E ratio, etc.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        fields = {
            "Company": info.get("longName", ticker.upper()),
            "Current Price": f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}",
            "Market Cap": f"${info.get('marketCap', 'N/A'):,}" if info.get("marketCap") else "N/A",
            "P/E Ratio (TTM)": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Dividend Yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "N/A",
            "Revenue (TTM)": f"${info.get('totalRevenue', 'N/A'):,}" if info.get("totalRevenue") else "N/A",
            "Profit Margin": f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get("profitMargins") else "N/A",
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
        }

        lines = [f"📈 Stock Info: {ticker.upper()}"]
        for k, v in fields.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    except Exception as e:
        logger.error("Stock data fetch failed for '{}': {}", ticker, e)
        return f"Could not retrieve data for ticker '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# Document Summary
# ---------------------------------------------------------------------------

@tool
def document_summary_tool(context: str) -> str:
    """
    Produce a concise bullet-point summary of the provided financial context.
    Use this after retrieving document chunks to distil the key information.

    Args:
        context: The raw text to summarise (typically retrieved document chunks).

    Returns:
        A structured summary string.
    """
    # This tool intentionally returns the context with a header so the
    # LLM can synthesise a summary as part of its reasoning step.
    if not context or not context.strip():
        return "No context provided for summarisation."
    return (
        "CONTEXT FOR SUMMARISATION:\n"
        + "=" * 60 + "\n"
        + context.strip()
        + "\n" + "=" * 60
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    rag_retrieval_tool,
    financial_calculator_tool,
    stock_data_tool,
    document_summary_tool,
]
