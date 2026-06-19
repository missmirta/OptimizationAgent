"""
Transportation Problem Solver
------------------------------
Deterministic solver - LLM does not come here.
Receives structured parameters, returns a structured result.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpInteger,
    lpSum, LpStatus, value, PULP_CBC_CMD
)


class SolverStatus(StrEnum):
    OPTIMAL = "Optimal"
    INFEASIBLE = "Infeasible"
    UNBOUNDED = "Unbounded"
    UNDEFINED = "Undefined"
    NOT_SOLVED = "Not Solved"


# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------

@dataclass
class TransportationParams:
    """
    Params of transport task.
    supply
    demand
    costs   — costs[warehouse][destination]
    """
    supply: dict[str, int]
    demand: dict[str, int]
    costs: dict[str, dict[str, int | float]]


@dataclass
class TransportationResult:
    status: SolverStatus
    total_cost: Optional[float]
    shipments: dict[str, dict[str, float]] # shipments[w][d]
    active_constraints: list[str]          # what restrictions
    insights: list[str]                    # human insights


# ---------------------------------------------------------------------------
# Balancing (dummy nodes)
# ---------------------------------------------------------------------------

def _balance(params: TransportationParams) -> TransportationParams:
    """
    If supply != demand — add dummy node.
    supply > demand → fiction point of destination
    demand > supply → fiction warehouse
    Return balanced parameters.
    """
    total_supply = sum(params.supply.values())
    total_demand = sum(params.demand.values())

    if total_supply == total_demand:
        return params

    supply = dict(params.supply)
    demand = dict(params.demand)
    costs = {w: dict(params.costs[w]) for w in params.costs}

    if total_supply > total_demand:
        diff = total_supply - total_demand
        dummy = "_DUMMY_DEST_"
        demand[dummy] = diff
        for w in supply:
            costs[w][dummy] = 0  # free saving
    else:
        diff = total_demand - total_supply
        dummy = "_DUMMY_SOURCE_"
        supply[dummy] = diff
        for d in demand:
            costs[dummy] = costs.get(dummy, {})
            costs[dummy][d] = 0

    return TransportationParams(supply=supply, demand=demand, costs=costs)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_transportation(params: TransportationParams) -> TransportationResult:
    """
    1. Balancing task
    2. Build PuLP model
    3. Resolve
    4. Returns structured result
    """
    balanced = _balance(params)

    warehouses = list(balanced.supply.keys())
    destinations = list(balanced.demand.keys())

    prob = LpProblem("Transportation", LpMinimize)

    # Variable solutions: how much to transport from each warehouse to each point
    routes = {
        (w, d): LpVariable(f"Route_{w}_{d}", lowBound=0, cat=LpInteger)
        for w in warehouses
        for d in destinations
    }

    # Objective function: minimize total cost
    prob += lpSum(
        balanced.costs[w][d] * routes[(w, d)]
        for w in warehouses
        for d in destinations
    ), "Total_Cost"

    # Limitation supply: from the warehouse no more than there is
    for w in warehouses:
        prob += (
            lpSum(routes[(w, d)] for d in destinations) <= balanced.supply[w],
            f"Supply_{w}"
        )

    # Limitation demand: to the point no less than ordered
    for d in destinations:
        prob += (
            lpSum(routes[(w, d)] for w in warehouses) >= balanced.demand[d],
            f"Demand_{d}"
        )

    # Resolving:
    prob.solve(PULP_CBC_CMD(msg=0))

    status = SolverStatus(LpStatus[prob.status])

    if status != SolverStatus.OPTIMAL:
        return TransportationResult(
            status=status,
            total_cost=None,
            shipments={},
            active_constraints=[],
            insights=[f"The problem has no solution.: {status}"]
        )

    # Collect the result (only real routes, no dummy)
    shipments: dict[str, dict[str, float]] = {}
    for w in params.supply:          # only original warehouses
        for d in params.demand:      # only original bars
            qty = routes[(w, d)].varValue or 0.0
            if qty > 0:
                shipments.setdefault(w, {})[d] = qty

    total_cost = value(prob.objective)

    # We define active restrictions (shadow price != 0 or binding)
    active = []
    for name, constraint in prob.constraints.items():
        if "DUMMY" in name:
            continue
        slack = constraint.slack
        if slack is not None and abs(slack) < 1e-6:
            active.append(name)

    insights = _generate_insights(params, shipments, total_cost, active)

    return TransportationResult(
        status=status,
        total_cost=total_cost,
        shipments=shipments,
        active_constraints=active,
        insights=insights
    )


# ---------------------------------------------------------------------------
# Insights (deterministic logic, no LLM)
# ---------------------------------------------------------------------------

def _generate_insights(
    params: TransportationParams,
    shipments: dict,
    total_cost: float,
    active: list[str]
) -> list[str]:
    insights = []

    # Which warehouses are completely exhausted?
    for w in params.supply:
        shipped = sum(shipments.get(w, {}).values())
        if abs(shipped - params.supply[w]) < 1e-6:
            insights.append(
                f"Warehouse {w} is fully exhausted ({int(shipped)} items.)"
            )
        elif shipped < params.supply[w]:
            leftover = params.supply[w] - shipped
            insights.append(
                f"Warehouse {w} has leftovers {int(leftover)} items. — not needed to transport further"
            )

    # The most expensive route that is still used
    used_costs = []
    for w, dests in shipments.items():
        for d, qty in dests.items():
            cost_per_unit = params.costs[w][d]
            used_costs.append((cost_per_unit, w, d, int(qty)))

    if used_costs:
        max_cost = max(used_costs, key=lambda x: x[0])
        insights.append(
            f"The most expensive Active route: {max_cost[1]}→{max_cost[2]} "
            f"(${max_cost[0]}/items., {max_cost[3]} items.) — forced due to demand constraints"
        )

    # Active supply/demand constraints
    supply_active = [c for c in active if c.startswith("Supply_")]
    demand_active = [c for c in active if c.startswith("Demand_")]

    if demand_active:
        points = [c.replace("Demand_", "") for c in demand_active]
        insights.append(
            f"Demand is met exactly (no surplus) in: {', '.join(points)}"
        )

    return insights


# ---------------------------------------------------------------------------
# Formated result
# ---------------------------------------------------------------------------

def format_result(params: TransportationParams, result: TransportationResult) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append(f"Status: {result.status}")
    lines.append("=" * 50)

    if result.status != SolverStatus.OPTIMAL:
        lines.append(result.insights[0] if result.insights else "")
        return "\n".join(lines)

    lines.append(f"\nTotal cost: ${result.total_cost:.2f}")
    lines.append("\nOptimal shipments:")
    lines.append("-" * 35)

    for w, dests in result.shipments.items():
        for d, qty in dests.items():
            cost = params.costs[w][d]
            lines.append(f"  {w} → {d}: {int(qty)} од.  (${cost}/од.)")

    lines.append("\nInsights:")
    for ins in result.insights:
        lines.append(f"  • {ins}")

    lines.append("=" * 50)
    return "\n".join(lines)
