"""
Shared state threaded through every LangGraph node.
All fields are optional so nodes can update only what they own.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────
    # Full message log (system / human / assistant); add_messages
    # reducer appends instead of overwriting.
    messages: Annotated[list, add_messages]

    # ── Classification ────────────────────────────────────────────
    problem_type: Optional[str]          # "transportation" | "blending" | "unknown"
    classification_confidence: Optional[str]
    classification_reason: Optional[str]

    # ── Extraction ────────────────────────────────────────────────
    # Raw extracted params stored as a plain dict so both problem
    # types can live in the same state without a union type.
    extracted_params: Optional[dict]
    is_complete: Optional[bool]
    missing_fields: Optional[list]
    follow_up_questions: Optional[list]

    # ── Model summary (shown to user before solving) ──────────────
    model_summary: Optional[str]

    # ── Approval ──────────────────────────────────────────────────
    approved: Optional[bool]

    # ── Solution ──────────────────────────────────────────────────
    solution: Optional[dict]        # serializable snapshot of solver result
    explanation: Optional[str]      # final human-language answer

    # ── Error ─────────────────────────────────────────────────────
    followup_attempts: Optional[int]   # incremented each time ask_followup fires

    # ── Error ─────────────────────────────────────────────────────────────────
    error: Optional[str]
