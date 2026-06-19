"""
Blending Problem Solver
-----------------------
Deterministic solver — LLM does not come here.
Receives structured parameters (BlendingProblemInput), returns a structured result.

Classic example: Whiskas cat food (PuLP documentation).
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from pulp import (
    LpProblem, LpMinimize, LpVariable,
    lpSum, LpStatus, value, PULP_CBC_CMD
)

from schemas.blending import BlendingProblemInput, NutrientRequirement


class SolverStatus(StrEnum):
    OPTIMAL = "Optimal"
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    UNDEFINED = "Undefined"
    NOT_SOLVED = "Not Solved"


@dataclass
class BlendingResult:
    status: SolverStatus
    total_cost: Optional[float]
    mix: dict[str, float]           # ingredient → grams (or units)
    mix_pct: dict[str, float]       # ingredient → percentage of total
    active_constraints: list[str]
    insights: list[str]


def solve_blending(params: BlendingProblemInput) -> BlendingResult:
    """
    Build and solve the blending LP:
      minimise  sum(cost_i * x_i)
      subject to
        nutritional constraints  (>=, <=, ==)
        sum(x_i) == total
        x_i >= 0
    """
    if not params.is_complete():
        missing = params.missing_fields()
        return BlendingResult(
            status=SolverStatus.NOT_SOLVED,
            total_cost=None,
            mix={},
            mix_pct={},
            active_constraints=[],
            insights=[f"Cannot solve — missing: {', '.join(missing)}"],
        )

    ingredients = params.ingredients
    prob = LpProblem("Blending", LpMinimize)

    # Decision variables: how many grams of each ingredient
    x = {
        ing: LpVariable(f"x_{ing}", lowBound=0)
        for ing in ingredients
    }

    # Objective: minimise total cost
    prob += lpSum(params.costs[ing] * x[ing] for ing in ingredients), "Total_Cost"

    # Total quantity constraint
    prob += (
        lpSum(x[ing] for ing in ingredients) == params.total,
        "Total_Quantity"
    )

    # Nutritional constraints
    for nutrient, req in params.requirements.items():
        nutrient_expr = lpSum(
            params.nutrients[nutrient][ing] * x[ing]
            for ing in ingredients
        )
        op = req.operator
        if op == ">=":
            prob += (nutrient_expr >= req.value, f"Nutrient_{nutrient}_min")
        elif op == "<=":
            prob += (nutrient_expr <= req.value, f"Nutrient_{nutrient}_max")
        else:  # ==
            prob += (nutrient_expr == req.value, f"Nutrient_{nutrient}_eq")

    prob.solve(PULP_CBC_CMD(msg=0))

    status = SolverStatus(LpStatus[prob.status])

    if status != SolverStatus.OPTIMAL:
        return BlendingResult(
            status=status,
            total_cost=None,
            mix={},
            mix_pct={},
            active_constraints=[],
            insights=[f"No optimal solution found: {status}"],
        )

    mix = {ing: round(x[ing].varValue or 0.0, 4) for ing in ingredients}
    total_cost = round(value(prob.objective), 6)
    total_qty = params.total

    mix_pct = {
        ing: round(100.0 * qty / total_qty, 2)
        for ing, qty in mix.items()
    }

    # Active constraints (slack ≈ 0)
    active = [
        name for name, con in prob.constraints.items()
        if con.slack is not None and abs(con.slack) < 1e-6
        and name != "Total_Quantity"
    ]

    insights = _generate_insights(params, mix, mix_pct, total_cost, active)

    return BlendingResult(
        status=status,
        total_cost=total_cost,
        mix=mix,
        mix_pct=mix_pct,
        active_constraints=active,
        insights=insights,
    )


def _generate_insights(
    params: BlendingProblemInput,
    mix: dict[str, float],
    mix_pct: dict[str, float],
    total_cost: float,
    active: list[str],
) -> list[str]:
    insights = []

    # Dominant ingredient
    dominant = max(mix, key=lambda i: mix[i])
    insights.append(
        f"Dominant ingredient: {dominant} ({mix_pct[dominant]:.1f}%) — "
        f"cheapest at ${params.costs[dominant]}/unit"
        if params.costs[dominant] == min(params.costs.values())
        else f"Dominant ingredient: {dominant} ({mix_pct[dominant]:.1f}%)"
    )

    # Which nutritional constraints are binding
    for name in active:
        if name.startswith("Nutrient_"):
            parts = name.split("_")
            nutrient = parts[1]
            direction = "minimum" if name.endswith("_min") else "maximum"
            req = params.requirements.get(nutrient)
            if req:
                insights.append(
                    f"Binding constraint: {nutrient} {req.operator} {req.value} "
                    f"({direction}) — this limits how much cheap filler can be used"
                )

    # Cost per unit of total
    cost_per_unit = total_cost / params.total if params.total else 0
    insights.append(
        f"Total cost: ${total_cost:.4f} for {params.total:.0f} units "
        f"(${cost_per_unit:.4f} per unit)"
    )

    return insights


def format_result(params: BlendingProblemInput, result: BlendingResult) -> str:
    lines = ["=" * 50, f"Status: {result.status}", "=" * 50]

    if result.status != SolverStatus.OPTIMAL:
        lines.append(result.insights[0] if result.insights else "")
        return "\n".join(lines)

    lines.append(f"\nOptimal blend (total: {params.total:.0f} units):")
    lines.append("-" * 35)
    for ing in params.ingredients:
        qty = result.mix.get(ing, 0)
        pct = result.mix_pct.get(ing, 0)
        cost = params.costs[ing]
        lines.append(f"  {ing:15s}: {qty:7.2f}  ({pct:5.1f}%)   @ ${cost}/unit")

    lines.append(f"\nTotal cost: ${result.total_cost:.4f}")
    lines.append("\nInsights:")
    for ins in result.insights:
        lines.append(f"  • {ins}")
    lines.append("=" * 50)
    return "\n".join(lines)
