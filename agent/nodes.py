"""
LangGraph node functions.
Each node receives the full AgentState and returns a partial update dict.
LLM calls happen only in classify / extract.  All math is in the solvers.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import interrupt

from extractor.classifier import classify, ProblemType
from extractor.llm_extractor import (
    extract_transportation_params,
    extract_blending_params,
    TransportationExtractionResult,
    BlendingExtractionResult,
)
from schemas.transportation import TransportationProblemInput
from schemas.blending import BlendingProblemInput, NutrientRequirement
from solvers.transportation import TransportationParams, solve_transportation
from solvers.blending import solve_blending


# ── helpers ──────────────────────────────────────────────────────────────────

def _conversation_text(state: dict) -> str:
    """Flatten all HumanMessage content into one block for (re-)extraction."""
    parts = []
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            parts.append(str(m.content))
    return "\n\n---\n\n".join(parts)


# ── node: classify ────────────────────────────────────────────────────────────

def node_classify(state: dict) -> dict:
    text = _conversation_text(state)
    problem_type, confidence, reason = classify(text)
    return {
        "problem_type": problem_type.value,
        "classification_confidence": confidence,
        "classification_reason": reason,
    }


# ── node: extract ─────────────────────────────────────────────────────────────

def node_extract(state: dict) -> dict:
    text = _conversation_text(state)
    ptype = state.get("problem_type")

    if ptype == ProblemType.TRANSPORTATION:
        result: TransportationExtractionResult = extract_transportation_params(text)
        raw = {
            "supply": result.params.supply,
            "demand": result.params.demand,
            "costs":  result.params.costs,
        }
        return {
            "extracted_params":     raw,
            "is_complete":          result.is_complete,
            "missing_fields":       result.missing_fields,
            "follow_up_questions":  result.follow_up_questions,
        }

    elif ptype == ProblemType.BLENDING:
        result: BlendingExtractionResult = extract_blending_params(text)
        # Serialise NutrientRequirement objects → plain dicts
        reqs = {}
        if result.params.requirements:
            reqs = {
                k: {"operator": v.operator, "value": v.value}
                for k, v in result.params.requirements.items()
            }
        raw = {
            "ingredients": result.params.ingredients,
            "costs":       result.params.costs,
            "nutrients":   result.params.nutrients,
            "requirements": reqs,
            "total":       result.params.total,
        }
        return {
            "extracted_params":     raw,
            "is_complete":          result.is_complete,
            "missing_fields":       result.missing_fields,
            "follow_up_questions":  result.follow_up_questions,
        }

    return {"error": f"Cannot extract: unknown problem type '{ptype}'"}


# ── node: ask follow-up ───────────────────────────────────────────────────────

def node_ask_followup(state: dict) -> dict:
    """
    Human-in-the-loop: show missing-field questions, wait for user answer.
    LangGraph interrupt() pauses execution; caller resumes with user text.
    """
    questions = state.get("follow_up_questions") or []
    prompt_lines = ["I need a bit more information to set up the model:"]
    for i, q in enumerate(questions, 1):
        prompt_lines.append(f"  {i}. {q}")
    prompt_lines.append("\nPlease answer the questions above.")
    prompt_text = "\n".join(prompt_lines)

    # Surface the questions to the UI
    answer = interrupt({"node": "ask_followup", "question": prompt_text})

    return {
        "followup_attempts": state.get("followup_attempts", 0) + 1,
        "messages": [
            AIMessage(content=prompt_text),
            HumanMessage(content=answer),
        ]
    }


# ── node: build model summary ─────────────────────────────────────────────────

def node_build_model_summary(state: dict) -> dict:
    """Produce a human-readable description of the optimisation model."""
    ptype  = state.get("problem_type")
    params = state.get("extracted_params") or {}
    lines  = []

    if ptype == "transportation":
        lines.append("Transportation Problem — minimise total shipping cost\n")
        supply = params.get("supply") or {}
        demand = params.get("demand") or {}
        costs  = params.get("costs")  or {}

        lines.append("Supply (warehouses):")
        for w, q in supply.items():
            lines.append(f"  - {w}: {q} units")

        lines.append("\nDemand (destinations):")
        for d, q in demand.items():
            lines.append(f"  - {d}: {q} units")

        lines.append("\nShipping costs (\\$/unit):")
        for w, dests in costs.items():
            row = ",  ".join(f"{d}: \\${c}" for d, c in dests.items())
            lines.append(f"  • {w} → {row}")

        total_supply = sum(supply.values())
        total_demand = sum(demand.values())
        if total_supply != total_demand:
            lines.append(
                f"\n Supply ({total_supply}) ≠ Demand ({total_demand})"
                " — a dummy node will be added automatically."
            )

    elif ptype == "blending":
        total = params.get("total", "?")
        lines.append(f"Blending Problem — minimise cost per {total} units\n")

        ingredients = params.get("ingredients") or []
        costs       = params.get("costs") or {}
        nutrients   = params.get("nutrients") or {}
        requirements = params.get("requirements") or {}

        lines.append("Ingredients:")
        for ing in ingredients:
            c = costs.get(ing, "?")
            lines.append(f"  • {ing}: \\${c}/unit")

        lines.append("\nNutritional constraints:")
        for nutrient, req in requirements.items():
            op  = req["operator"] if isinstance(req, dict) else req.operator
            val = req["value"]    if isinstance(req, dict) else req.value
            lines.append(f"  • {nutrient} {op} {val}")

    summary = "\n".join(lines)
    return {"model_summary": summary}


# ── node: approve ─────────────────────────────────────────────────────────────

def node_approve(state: dict) -> dict:
    """
    Human-in-the-loop: show model summary and ask for approval.
    'yes' / 'y' / 'ok' → approved; anything else → treated as a correction.
    """
    summary = state.get("model_summary", "")
    prompt = (
        f"Here is the optimisation model I've built:\n\n"
        f"{summary}\n\n"
        "Type  yes  to solve it, or describe any corrections you'd like to make."
    )

    answer: str = interrupt({"node": "approve", "question": prompt, "summary": summary})
    approved = answer.strip().lower() in {"yes", "y", "ok", "go", "proceed", "solve"}

    if approved:
        return {
            "approved": True,
            "messages": [AIMessage(content=prompt), HumanMessage(content=answer)],
        }
    else:
        # Feed correction back into the conversation so node_extract re-reads it
        return {
            "approved": False,
            "messages": [AIMessage(content=prompt), HumanMessage(content=answer)],
        }


# ── node: solve ───────────────────────────────────────────────────────────────

def node_solve(state: dict) -> dict:
    ptype  = state.get("problem_type")
    params = state.get("extracted_params") or {}

    if ptype == "transportation":
        tp = TransportationParams(
            supply=params["supply"],
            demand=params["demand"],
            costs=params["costs"],
        )
        result = solve_transportation(tp)
        solution = {
            "status":             str(result.status),
            "total_cost":         result.total_cost,
            "shipments":          result.shipments,
            "active_constraints": result.active_constraints,
            "insights":           result.insights,
        }

    elif ptype == "blending":
        reqs = {
            k: NutrientRequirement(**v) if isinstance(v, dict) else v
            for k, v in params.get("requirements", {}).items()
        }
        bp = BlendingProblemInput(
            ingredients=params["ingredients"],
            costs=params["costs"],
            nutrients=params["nutrients"],
            requirements=reqs,
            total=params["total"],
        )
        result = solve_blending(bp)
        solution = {
            "status":             str(result.status),
            "total_cost":         result.total_cost,
            "mix":                result.mix,
            "mix_pct":            result.mix_pct,
            "active_constraints": result.active_constraints,
            "insights":           result.insights,
        }

    else:
        return {"error": f"Cannot solve unknown problem type '{ptype}'"}

    return {"solution": solution}


# ── node: explain ─────────────────────────────────────────────────────────────

def node_explain(state: dict) -> dict:
    """Format the solver result as a user-facing answer."""
    ptype    = state.get("problem_type")
    solution = state.get("solution") or {}
    params   = state.get("extracted_params") or {}
    lines    = []

    status = solution.get("status", "Unknown")
    if status != "Optimal":
        explanation = f"The solver could not find an optimal solution (status: {status})."
        return {
            "explanation": explanation,
            "messages": [AIMessage(content=explanation)],
        }

    if ptype == "transportation":
        cost = solution.get("total_cost", 0)
        lines.append(f"Optimal solution found — total shipping cost: **\\${cost:.2f}**\n")
        lines.append("Shipments:")
        for w, dests in (solution.get("shipments") or {}).items():
            for d, qty in dests.items():
                c = params.get("costs", {}).get(w, {}).get(d, "?")
                lines.append(f"  • {w} → {d}: {int(qty)} units  @ \\${c}/unit")

    elif ptype == "blending":
        cost  = solution.get("total_cost", 0)
        total = params.get("total", 100)
        lines.append(f"Optimal blend found — total cost: **\\${cost:.4f}** per {total:.0f} units\n")
        lines.append("Mix:")
        for ing, qty in (solution.get("mix") or {}).items():
            pct = solution.get("mix_pct", {}).get(ing, 0)
            c   = (params.get("costs") or {}).get(ing, "?")
            lines.append(f"  • {ing}: {qty:.1f} units  ({pct:.1f}%)  @ \\${c}/unit")

    lines.append("\nKey insights:")
    for ins in (solution.get("insights") or []):
        lines.append(f"  • {ins}")

    explanation = "\n".join(lines)
    return {
        "explanation": explanation,
        "messages": [AIMessage(content=explanation)],
    }
