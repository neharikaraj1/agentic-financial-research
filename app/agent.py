"""
LangGraph Agentic Workflow for Financial Research.

Graph nodes:
  1. query_planner   – Break the user query into a structured plan
  2. retriever       – Fetch relevant document chunks from ChromaDB (RAG)
  3. tool_executor   – Execute any requested tools (calculator, stock data)
  4. reasoner        – Synthesise evidence into a reasoned answer draft
  5. validator       – Check the answer for factual grounding and completeness
  6. response_gen    – Format the final grounded response

Edges use conditional routing so the agent can loop back or terminate early.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.tools import BaseTool
from langgraph.graph import END, StateGraph
from loguru import logger

from app.config import get_settings
from app.tools import ALL_TOOLS, rag_retrieval_tool, financial_calculator_tool, stock_data_tool


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Shared mutable state passed between graph nodes."""
    original_query: str              # Raw user question
    plan: List[str]                  # Ordered list of reasoning steps
    retrieved_context: str           # Concatenated RAG passages
    tool_outputs: List[Dict[str, Any]]  # Results from tool calls
    reasoning: str                   # Internal reasoning draft
    validation_passed: bool          # Whether validator approved the draft
    validation_notes: str            # Feedback from validator node
    final_answer: str                # The polished final response
    iteration: int                   # Loop counter (guards against infinite loops)
    sources: List[str]               # Source documents referenced


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.0) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        openai_api_key=settings.openai_api_key,
    )


# ---------------------------------------------------------------------------
# Node 1 – Query Planner
# ---------------------------------------------------------------------------

def query_planner_node(state: AgentState) -> AgentState:
    """
    Decompose the user's financial query into a numbered action plan.
    The plan guides subsequent nodes on what to retrieve and compute.
    """
    logger.info("[Node: query_planner] Planning for: '{}'", state["original_query"])
    llm = _get_llm()

    system = SystemMessage(content="""You are a senior financial research analyst.
Your job is to create a concise, step-by-step research plan to answer the user's financial query.

Output ONLY a JSON array of strings, each representing one action step. Example:
["Retrieve relevant financial statements from documents", "Calculate P/E ratio", "Fetch live stock price", "Synthesise findings"]

Keep the plan to 3-6 steps. Be specific about what data needs to be retrieved or calculated.""")

    human = HumanMessage(content=f"Create a research plan for: {state['original_query']}")

    response = llm.invoke([system, human])
    raw = response.content.strip()

    # Robustly parse the JSON plan
    try:
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        plan: List[str] = json.loads(raw)
        if not isinstance(plan, list):
            raise ValueError("Plan is not a list")
    except Exception:
        # Fallback: treat each line as a step
        plan = [line.strip("- ").strip() for line in raw.split("\n") if line.strip()]

    logger.info("[Node: query_planner] Plan: {}", plan)
    return {
        **state,
        "plan": plan,
        "iteration": state.get("iteration", 0),
        "tool_outputs": [],
        "sources": [],
    }


# ---------------------------------------------------------------------------
# Node 2 – Retriever (RAG)
# ---------------------------------------------------------------------------

def retriever_node(state: AgentState) -> AgentState:
    """
    Run semantic similarity search against ChromaDB using the original query
    plus key terms extracted from the plan.
    """
    logger.info("[Node: retriever] Running RAG retrieval")
    query = state["original_query"]
    plan_str = " ".join(state.get("plan", []))

    # Enhanced query combines original question with plan context
    enhanced_query = f"{query}\n\nResearch focus: {plan_str}"
    context = rag_retrieval_tool.invoke({"query": enhanced_query, "k": 6})

    # Extract source names
    sources: List[str] = []
    for line in context.split("\n"):
        if line.startswith("Source:") or "Source:" in line:
            parts = line.split("|")
            for part in parts:
                if "Source:" in part:
                    source = part.replace("Source:", "").strip()
                    if source not in sources:
                        sources.append(source)

    logger.info("[Node: retriever] Retrieved {} chars of context", len(context))
    return {**state, "retrieved_context": context, "sources": sources}


# ---------------------------------------------------------------------------
# Node 3 – Tool Executor
# ---------------------------------------------------------------------------

def tool_executor_node(state: AgentState) -> AgentState:
    """
    Use the LLM to decide which tools to call based on the plan,
    then execute those tools and collect their outputs.
    """
    logger.info("[Node: tool_executor] Deciding tool calls")
    llm = _get_llm()

    tool_descriptions = "\n".join([
        f"- {t.name}: {t.description.split(chr(10))[0]}"
        for t in ALL_TOOLS
    ])

    system = SystemMessage(content=f"""You are a financial research tool orchestrator.
Based on the research plan and query, decide which tools to call.

Available tools:
{tool_descriptions}

Output a JSON array of tool calls. Each call: {{"tool": "tool_name", "input": {{...}}}}
If no tools are needed beyond RAG, output: []

Only call stock_data_tool if a specific ticker is mentioned.
Only call financial_calculator_tool if specific numbers are provided.""")

    human = HumanMessage(content=f"""
Query: {state['original_query']}
Plan: {json.dumps(state.get('plan', []))}
Retrieved context (first 500 chars): {state.get('retrieved_context', '')[:500]}
""")

    response = llm.invoke([system, human])
    raw = response.content.strip()

    tool_outputs: List[Dict[str, Any]] = list(state.get("tool_outputs", []))

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        calls = json.loads(raw)
        if not isinstance(calls, list):
            calls = []
    except Exception:
        calls = []

    for call in calls:
        tool_name = call.get("tool", "")
        tool_input = call.get("input", {})
        logger.info("[Node: tool_executor] Calling tool '{}' with {}", tool_name, tool_input)

        tool_map = {t.name: t for t in ALL_TOOLS}
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_input)
                tool_outputs.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": result,
                })
            except Exception as e:
                tool_outputs.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": f"Error: {str(e)}",
                })

    return {**state, "tool_outputs": tool_outputs}


# ---------------------------------------------------------------------------
# Node 4 – Reasoner
# ---------------------------------------------------------------------------

def reasoner_node(state: AgentState) -> AgentState:
    """
    Synthesise retrieved context and tool outputs into a draft answer.
    Explicit chain-of-thought reasoning is applied here.
    """
    logger.info("[Node: reasoner] Synthesising answer")
    llm = _get_llm(temperature=0.1)

    tool_summary = ""
    for t in state.get("tool_outputs", []):
        tool_summary += f"\n[{t['tool']}] {t['output']}\n"

    system = SystemMessage(content="""You are an expert financial research analyst.
Your task is to synthesise the retrieved document context and tool results into a comprehensive, grounded answer.

Guidelines:
- Only make claims supported by the retrieved evidence
- Cite source documents when making specific claims (e.g., "According to [source]...")
- Show calculations clearly when relevant
- Identify any gaps or uncertainties in the available information
- Structure your answer clearly with sections if needed""")

    human = HumanMessage(content=f"""
ORIGINAL QUERY: {state['original_query']}

RESEARCH PLAN:
{chr(10).join(f"  {i+1}. {s}" for i, s in enumerate(state.get('plan', [])))}

RETRIEVED DOCUMENT CONTEXT:
{state.get('retrieved_context', 'No documents retrieved.')}

TOOL OUTPUTS:
{tool_summary if tool_summary else 'No additional tool outputs.'}

Please synthesise the above into a thorough, grounded answer.
""")

    response = llm.invoke([system, human])
    reasoning = response.content.strip()
    logger.info("[Node: reasoner] Draft answer length: {} chars", len(reasoning))
    return {**state, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Node 5 – Validator
# ---------------------------------------------------------------------------

def validator_node(state: AgentState) -> AgentState:
    """
    Critically review the reasoner's draft.
    Checks: factual grounding, completeness, any hallucinations.
    """
    logger.info("[Node: validator] Validating draft answer")
    llm = _get_llm()

    system = SystemMessage(content="""You are a rigorous financial fact-checker.
Review the draft answer against the source context. Check for:
1. Factual accuracy — are all claims supported by the context?
2. Completeness — does it address all parts of the query?
3. Hallucinations — any invented numbers or unsupported statements?
4. Clarity — is it easy to understand?

Output JSON: {"passed": true/false, "notes": "feedback here"}
Set passed=true only if the answer is factually grounded and complete.""")

    human = HumanMessage(content=f"""
QUERY: {state['original_query']}
SOURCE CONTEXT (first 1500 chars): {state.get('retrieved_context', '')[:1500]}
DRAFT ANSWER: {state.get('reasoning', '')}
""")

    response = llm.invoke([system, human])
    raw = response.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        result = json.loads(raw)
        passed = bool(result.get("passed", True))
        notes = result.get("notes", "")
    except Exception:
        passed = True
        notes = "Could not parse validation response — assuming passed."

    logger.info("[Node: validator] passed={}, notes={}", passed, notes)
    return {
        **state,
        "validation_passed": passed,
        "validation_notes": notes,
        "iteration": state.get("iteration", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node 6 – Response Generator
# ---------------------------------------------------------------------------

def response_gen_node(state: AgentState) -> AgentState:
    """
    Polish the validated answer into a final, well-formatted response
    with source attribution.
    """
    logger.info("[Node: response_gen] Generating final response")
    llm = _get_llm(temperature=0.2)

    validation_feedback = state.get("validation_notes", "")
    if not state.get("validation_passed", True) and validation_feedback:
        improvement_instruction = f"\nIMPROVEMENT NEEDED: {validation_feedback}"
    else:
        improvement_instruction = ""

    system = SystemMessage(content="""You are a professional financial research analyst.
Format the final answer in clean, readable markdown.
Include:
- A concise executive summary at the top
- Well-structured sections
- Source citations where applicable
- A 'Data Sources' section at the bottom listing documents used
- Any caveats or data limitations
Use **bold** for key figures and findings.""")

    human = HumanMessage(content=f"""
QUERY: {state['original_query']}
DRAFT ANSWER: {state.get('reasoning', '')}
SOURCES USED: {', '.join(state.get('sources', [])) or 'None specified'}
{improvement_instruction}

Format this into a final polished research response.
""")

    response = llm.invoke([system, human])
    final_answer = response.content.strip()
    logger.info("[Node: response_gen] Final answer ready ({} chars)", len(final_answer))
    return {**state, "final_answer": final_answer}


# ---------------------------------------------------------------------------
# Conditional edge – loop back or finish
# ---------------------------------------------------------------------------

def should_retry(state: AgentState) -> Literal["response_gen", "reasoner"]:
    """
    After validation, decide whether to regenerate or proceed to final output.
    Retries at most MAX_ITERATIONS times to prevent infinite loops.
    """
    max_iter = get_settings().max_iterations
    if state.get("validation_passed", True) or state.get("iteration", 0) >= max_iter:
        return "response_gen"
    logger.info("[Edge: should_retry] Validation failed — retrying reasoner (iter {})",
                state.get("iteration", 0))
    return "reasoner"


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """Compile and return the full LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("validator", validator_node)
    graph.add_node("response_gen", response_gen_node)

    # Define edges
    graph.set_entry_point("query_planner")
    graph.add_edge("query_planner", "retriever")
    graph.add_edge("retriever", "tool_executor")
    graph.add_edge("tool_executor", "reasoner")
    graph.add_edge("reasoner", "validator")
    graph.add_conditional_edges("validator", should_retry)
    graph.add_edge("response_gen", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

# Compile once at import time
_compiled_graph = None


def get_agent():
    """Return the compiled agent graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


def run_agent(query: str) -> Dict[str, Any]:
    """
    Run the full agentic pipeline for a financial query.

    Args:
        query: The user's financial research question.

    Returns:
        Dict with 'final_answer', 'plan', 'sources', 'tool_outputs', etc.
    """
    logger.info("Starting agent for query: '{}'", query)
    agent = get_agent()

    initial_state: AgentState = {
        "original_query": query,
        "plan": [],
        "retrieved_context": "",
        "tool_outputs": [],
        "reasoning": "",
        "validation_passed": False,
        "validation_notes": "",
        "final_answer": "",
        "iteration": 0,
        "sources": [],
    }

    result = agent.invoke(initial_state)
    logger.info("Agent completed. Final answer: {} chars", len(result.get("final_answer", "")))
    return result
