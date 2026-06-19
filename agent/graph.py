"""
LangGraph state machine for the Optimization Agent.

Pipeline:
  START
    → classify
    → extract
    → check_completeness ─── incomplete ──→ ask_followup ──→ extract
                         └── complete ───→ build_model_summary
                                              → approve ──── edit ──→ extract
                                                       └── yes ──→ solve
                                                                    → explain
                                                                       → END

Two human-in-the-loop interrupts:
  1. ask_followup  – agent asks clarifying questions
  2. approve       – agent shows model; user approves or requests corrections
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    node_classify,
    node_extract,
    node_ask_followup,
    node_build_model_summary,
    node_approve,
    node_solve,
    node_explain,
)


# ── routing functions ─────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    ptype = state.get("problem_type", "unknown")
    if ptype in ("transportation", "blending"):
        return "extract"
    return "unknown_end"


MAX_FOLLOWUP_ATTEMPTS = 5


def route_after_extract(state: AgentState) -> str:
    if state.get("error"):
        return "error_end"
    if state.get("is_complete"):
        return "build_model_summary"
    if (state.get("followup_attempts") or 0) >= MAX_FOLLOWUP_ATTEMPTS:
        return "max_attempts_end"
    return "ask_followup"


def route_after_approve(state: AgentState) -> str:
    if state.get("approved"):
        return "solve"
    # User gave a correction → re-extract with the updated conversation
    return "extract"


# ── graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("classify",           node_classify)
    g.add_node("extract",            node_extract)
    g.add_node("ask_followup",       node_ask_followup)
    g.add_node("build_model_summary", node_build_model_summary)
    g.add_node("approve",            node_approve)
    g.add_node("solve",              node_solve)
    g.add_node("explain",            node_explain)

    # Terminal nodes for dead-end paths
    g.add_node("unknown_end", lambda s: {
        "explanation": (
            "I couldn't recognise this as a transportation or blending problem. "
            "Please describe the problem in more detail — "
            "mention sources/destinations (transportation) or "
            "ingredients/nutritional requirements (blending)."
        )
    })
    g.add_node("error_end", lambda s: {
        "explanation": f"An error occurred: {s.get('error', 'unknown error')}"
    })
    g.add_node("max_attempts_end", lambda s: {
        "explanation": (
            f"I was unable to collect all required information after "
            f"{MAX_FOLLOWUP_ATTEMPTS} follow-up attempts. "
            "Please start a new problem and provide as much detail as possible upfront."
        )
    })

    # Edges
    g.add_edge(START, "classify")

    g.add_conditional_edges("classify", route_after_classify, {
        "extract":      "extract",
        "unknown_end":  "unknown_end",
    })

    g.add_conditional_edges("extract", route_after_extract, {
        "build_model_summary": "build_model_summary",
        "ask_followup":        "ask_followup",
        "error_end":           "error_end",
        "max_attempts_end":    "max_attempts_end",
    })

    # follow-up loop: after answering questions, re-extract
    g.add_edge("ask_followup", "extract")

    g.add_edge("build_model_summary", "approve")

    g.add_conditional_edges("approve", route_after_approve, {
        "solve":   "solve",
        "extract": "extract",   # edit loop
    })

    g.add_edge("solve", "explain")
    g.add_edge("explain",     END)
    g.add_edge("unknown_end",      END)
    g.add_edge("error_end",        END)
    g.add_edge("max_attempts_end", END)

    return g


def compile_graph():
    """Return a compiled graph with an in-memory checkpointer."""
    g = build_graph()
    memory = MemorySaver()
    return g.compile(checkpointer=memory)
