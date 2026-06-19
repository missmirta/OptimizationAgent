"""
LLM-powered parameter extractor (OpenAI backend).
Responsibility: understand user text → populate Pydantic schema → identify gaps.
The LLM never touches math; it only fills a JSON structure.
"""

import os
import json
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

from schemas.transportation import TransportationProblemInput
from schemas.blending import BlendingProblemInput, NutrientRequirement

load_dotenv()

_MODEL = "gpt-4o-mini"

_TRANSPORTATION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_transportation_params",
        "description": (
            "Extract transportation problem parameters from user text. "
            "Only include data explicitly mentioned. Omit (do not guess) any field "
            "whose value is not clearly stated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "supply": {
                    "type": "object",
                    "description": (
                        "Map of warehouse/source name → available supply quantity. "
                        'Example: {"Warehouse A": 1000, "Factory B": 4000}'
                    ),
                    "additionalProperties": {"type": "number"},
                },
                "demand": {
                    "type": "object",
                    "description": (
                        "Map of destination/customer name → required demand quantity. "
                        'Example: {"Store 1": 500, "City 2": 900}'
                    ),
                    "additionalProperties": {"type": "number"},
                },
                "costs": {
                    "type": "object",
                    "description": (
                        "Nested map: source → destination → cost per unit shipped. "
                        'Example: {"Warehouse A": {"Store 1": 2.0, "Store 2": 4.0}}'
                    ),
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
            },
            "required": [],
        },
    },
}

_BLENDING_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_blending_params",
        "description": (
            "Extract blending/mix optimisation problem parameters from user text. "
            "Only include data explicitly mentioned. Omit (do not guess) any field "
            "whose value is not clearly stated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of available ingredient names.",
                },
                "costs": {
                    "type": "object",
                    "description": "Map of ingredient name → cost per unit (gram, kg, etc.).",
                    "additionalProperties": {"type": "number"},
                },
                "nutrients": {
                    "type": "object",
                    "description": (
                        "Nested map: nutrient_name → {ingredient_name: fraction_or_value}. "
                        'Example: {"protein": {"chicken": 0.1, "beef": 0.2}}'
                    ),
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "requirements": {
                    "type": "object",
                    "description": (
                        "Nutritional constraints per nutrient. "
                        "Each value has 'operator' (>= / <= / ==) and 'value'. "
                        'Example: {"protein": {"operator": ">=", "value": 8.0}}'
                    ),
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "operator": {
                                "type": "string",
                                "enum": [">=", "<=", "=="],
                            },
                            "value": {"type": "number"},
                        },
                        "required": ["operator", "value"],
                    },
                },
                "total": {
                    "type": "number",
                    "description": "Total quantity of the blend (e.g. 100 for a 100g can).",
                },
            },
            "required": [],
        },
    },
}

_SYSTEM_PROMPT = """\
You are a parameter extraction assistant for an optimization problem solver.
Your only job is to read the user's description and call the extraction tool with
whatever parameters are explicitly mentioned.

Rules:
- Do NOT invent or infer values that are not stated.
- Do NOT include a field if its value is unclear or missing.
- Use the exact names the user gives for warehouses, stores, ingredients, etc.
- Numeric values must be numbers, not strings.
- If the text mentions nothing about costs, omit the costs field entirely.
- For nutrients: if a value is explicitly stated as 0 or 0.0, include it — zero is a valid value.
  Every ingredient mentioned in the nutritional table must appear in each nutrient dict,
  even when its value is 0.
"""


@dataclass
class TransportationExtractionResult:
    params: TransportationProblemInput
    is_complete: bool
    missing_fields: list[str]
    follow_up_questions: list[str]


@dataclass
class BlendingExtractionResult:
    params: BlendingProblemInput
    is_complete: bool
    missing_fields: list[str]
    follow_up_questions: list[str]


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or environment."
        )
    return OpenAI(api_key=api_key)


def extract_transportation_params(text: str) -> TransportationExtractionResult:
    """
    Send user text to GPT, extract transportation parameters via tool calling,
    validate with Pydantic, and return the result with any follow-up questions.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tools=[_TRANSPORTATION_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_transportation_params"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    raw: dict = json.loads(tool_call.function.arguments)

    params = TransportationProblemInput(**raw)

    return TransportationExtractionResult(
        params=params,
        is_complete=params.is_complete(),
        missing_fields=params.missing_fields() + [
            f"{w}→{d}" for w, d in params.missing_routes()
        ],
        follow_up_questions=params.follow_up_questions(),
    )


def extract_blending_params(text: str) -> BlendingExtractionResult:
    """
    Send user text to GPT, extract blending parameters via tool calling,
    validate with Pydantic, and return the result with any follow-up questions.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tools=[_BLENDING_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_blending_params"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    raw: dict = json.loads(tool_call.function.arguments)

    if "requirements" in raw and raw["requirements"]:
        raw["requirements"] = {
            k: NutrientRequirement(**v)
            for k, v in raw["requirements"].items()
        }

    params = BlendingProblemInput(**raw)

    return BlendingExtractionResult(
        params=params,
        is_complete=params.is_complete(),
        missing_fields=params.missing_fields(),
        follow_up_questions=params.follow_up_questions(),
    )
