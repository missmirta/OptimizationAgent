"""
Problem-type classifier.
Uses OpenAI with few-shot examples to return: transportation | blending | unknown.
LLM only outputs a label — no math, no parameters.
"""

import os
import json
from enum import StrEnum

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_MODEL = "gpt-4o-mini"


class ProblemType(StrEnum):
    TRANSPORTATION = "transportation"
    BLENDING = "blending"
    UNKNOWN = "unknown"


_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_problem",
        "description": "Classify the optimisation problem type described by the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "problem_type": {
                    "type": "string",
                    "enum": ["transportation", "blending", "unknown"],
                    "description": (
                        "transportation — moving goods from sources to destinations at minimum cost. "
                        "blending — mixing ingredients to meet nutritional/quality constraints at minimum cost. "
                        "unknown — neither type, or too vague to classify."
                    ),
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How confident the classification is.",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining why this type was chosen.",
                },
            },
            "required": ["problem_type", "confidence", "reason"],
        },
    },
}

_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "We have two factories in Kyiv and Lviv. "
            "We need to ship products to warehouses in Odesa, Kharkiv, and Dnipro. "
            "We want to minimise transport costs."
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "ex1",
            "type": "function",
            "function": {
                "name": "classify_problem",
                "arguments": json.dumps({
                    "problem_type": "transportation",
                    "confidence": "high",
                    "reason": "Moving goods from multiple sources to multiple destinations to minimise shipping cost.",
                }),
            },
        }],
    },
    {"role": "tool", "tool_call_id": "ex1", "content": "ok"},
    {
        "role": "user",
        "content": (
            "I want to make a 100g can of pet food using beef, chicken and gel. "
            "Each ingredient has a cost and a nutritional profile. "
            "The final mix must meet minimum protein and maximum fat requirements."
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "ex2",
            "type": "function",
            "function": {
                "name": "classify_problem",
                "arguments": json.dumps({
                    "problem_type": "blending",
                    "confidence": "high",
                    "reason": "Mixing ingredients to satisfy nutritional constraints at minimum cost — classic blending problem.",
                }),
            },
        }],
    },
    {"role": "tool", "tool_call_id": "ex2", "content": "ok"},
    {
        "role": "user",
        "content": "What is the fastest route from Kyiv to Berlin?",
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "ex3",
            "type": "function",
            "function": {
                "name": "classify_problem",
                "arguments": json.dumps({
                    "problem_type": "unknown",
                    "confidence": "high",
                    "reason": "Shortest-path query, not a transportation or blending optimisation problem.",
                }),
            },
        }],
    },
    {"role": "tool", "tool_call_id": "ex3", "content": "ok"},
]

_SYSTEM_PROMPT = """\
You are a classifier for optimisation problems.
Read the user's description and decide which category it belongs to:

- transportation: shipping/distribution from sources to destinations, minimise transport cost
- blending: mixing ingredients/materials to meet constraints (nutrition, quality, composition)
- unknown: anything else, or too vague to classify

Use the classify_problem tool to return your answer.
"""


def classify(text: str) -> tuple[ProblemType, str, str]:
    """
    Returns (problem_type, confidence, reason).
    Never raises — returns (UNKNOWN, "low", <error msg>) on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ProblemType.UNKNOWN, "low", "OPENAI_API_KEY not set"

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                *_FEW_SHOT,
                {"role": "user", "content": text},
            ],
            tools=[_CLASSIFY_TOOL],
            tool_choice={"type": "function", "function": {"name": "classify_problem"}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        raw = json.loads(tool_call.function.arguments)
        return ProblemType(raw["problem_type"]), raw["confidence"], raw["reason"]
    except Exception as exc:
        return ProblemType.UNKNOWN, "low", f"Classification error: {exc}"
