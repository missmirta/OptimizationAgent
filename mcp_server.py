"""
MCP Server for Optimization Agent
----------------------------------
Wraps the two deterministic solvers as MCP tools.
The solvers know nothing about MCP — this file is the only glue layer.

Transport: stdio (works with Claude Desktop out of the box).

Start:
    python mcp_server.py          # stdio (for Claude Desktop / LangGraph)
    mcp dev mcp_server.py         # MCP Inspector (browser UI for manual testing)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

from solvers.transportation import TransportationParams, solve_transportation
from solvers.blending import solve_blending
from schemas.blending import BlendingProblemInput, NutrientRequirement

mcp = FastMCP("optimization-agent")


# ── Tool 1: Transportation ────────────────────────────────────────────────────

@mcp.tool()
def solve_transportation_problem(
    supply: dict,
    demand: dict,
    costs: dict,
) -> dict:
    """Solve a transportation problem: minimise total shipping cost from
    warehouses to destinations.

    Call this tool when the user has:
    - sources (warehouses / factories) with a limited supply quantity
    - destinations (stores / customers) with a demand quantity
    - per-unit shipping costs between each source and each destination

    The solver automatically adds a dummy node if supply ≠ demand (unbalanced problem).
    It always returns an integer-valued optimal solution (transportation LP property).

    Args:
        supply: Map of warehouse name → available quantity.
                Example: {"A": 1000, "B": 4000}
        demand: Map of destination name → required quantity.
                Example: {"1": 500, "2": 900, "3": 1800}
        costs:  Nested map source → destination → cost per unit.
                Example: {"A": {"1": 2, "2": 4}, "B": {"1": 3, "2": 1}}

    Returns:
        On success: {"success": true, "status": "Optimal", "total_cost": ...,
                     "shipments": {...}, "insights": [...]}
        On failure: {"success": false, "status": ..., "error": ..., "hint": ...}
    """
    try:
        params = TransportationParams(
            supply={k: int(v) for k, v in supply.items()},
            demand={k: int(v) for k, v in demand.items()},
            costs={
                src: {dst: float(c) for dst, c in dests.items()}
                for src, dests in costs.items()
            },
        )
        result = solve_transportation(params)

        if str(result.status) != "Optimal":
            return {
                "success": False,
                "status": str(result.status),
                "error": f"No optimal solution: {result.status}",
                "hint": "Check that supply and demand values are positive and costs are provided for all routes.",
            }

        return {
            "success": True,
            "status": "Optimal",
            "total_cost": result.total_cost,
            "shipments": result.shipments,
            "active_constraints": result.active_constraints,
            "insights": result.insights,
        }

    except Exception as exc:
        return {
            "success": False,
            "status": "Error",
            "error": str(exc),
            "hint": "Verify supply/demand are dicts of {name: int} and costs is a nested dict {src: {dst: float}}.",
        }


# ── Tool 2: Blending ──────────────────────────────────────────────────────────

@mcp.tool()
def solve_blending_problem(
    ingredients: list,
    costs: dict,
    nutrients: dict,
    requirements: dict,
    total: float,
) -> dict:
    """Solve a blending problem: find the minimum-cost mixture of ingredients
    that satisfies nutritional or compositional constraints.

    Call this tool when the user has:
    - a list of ingredients with costs per unit
    - nutritional content per unit for each ingredient
    - minimum or maximum constraints on nutrient levels
    - a fixed total quantity to produce

    Args:
        ingredients: List of ingredient names.
                     Example: ["Chicken", "Beef", "Gel"]
        costs:       Map of ingredient → cost per unit (gram, kg, etc.).
                     Example: {"Beef": 0.008, "Gel": 0.001}
        nutrients:   Nested map nutrient → {ingredient: value_per_unit}.
                     Example: {"protein": {"Beef": 0.20, "Gel": 0.00},
                               "fat":     {"Beef": 0.10, "Gel": 0.00}}
        requirements: Map of nutrient → {operator, value}.
                      operator must be one of ">=", "<=", "==".
                      Example: {"protein": {"operator": ">=", "value": 8.0},
                                "fat":     {"operator": ">=", "value": 6.0}}
        total:       Total quantity of the blend to produce.
                     Example: 100.0  (for a 100g can)

    Returns:
        On success: {"success": true, "status": "Optimal", "total_cost": ...,
                     "composition": {"Beef": 60.0, "Gel": 40.0},
                     "composition_pct": {"Beef": 60.0, "Gel": 40.0},
                     "insights": [...]}
        On failure: {"success": false, "status": ..., "error": ..., "hint": ...}
    """
    try:
        reqs = {
            k: NutrientRequirement(operator=v["operator"], value=float(v["value"]))
            for k, v in requirements.items()
        }
        params = BlendingProblemInput(
            ingredients=ingredients,
            costs={k: float(v) for k, v in costs.items()},
            nutrients={
                nutrient: {ing: float(val) for ing, val in per_ing.items()}
                for nutrient, per_ing in nutrients.items()
            },
            requirements=reqs,
            total=float(total),
        )
        result = solve_blending(params)

        if str(result.status) != "Optimal":
            return {
                "success": False,
                "status": str(result.status),
                "error": f"No optimal solution: {result.status}",
                "hint": (
                    "The nutritional requirements may be impossible to satisfy "
                    "with the given ingredients. Check that at least one ingredient "
                    "contributes to each constrained nutrient."
                ),
            }

        return {
            "success": True,
            "status": "Optimal",
            "total_cost": result.total_cost,
            "composition": result.mix,
            "composition_pct": result.mix_pct,
            "active_constraints": result.active_constraints,
            "insights": result.insights,
        }

    except Exception as exc:
        return {
            "success": False,
            "status": "Error",
            "error": str(exc),
            "hint": (
                "Verify: ingredients is a list, costs/nutrients are dicts, "
                "requirements values have 'operator' and 'value' keys, total is a number."
            ),
        }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
