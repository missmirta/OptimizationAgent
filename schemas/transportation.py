"""
Pydantic schema for the Transportation problem.
Used for LLM extraction and completeness checking.
"""

from pydantic import BaseModel, model_validator
from typing import Optional


class TransportationProblemInput(BaseModel):
    """
    Partial or complete transportation problem — as extracted from user text.
    Fields are Optional so partial extraction is valid.
    """
    supply: Optional[dict[str, int]] = None
    demand: Optional[dict[str, int]] = None
    costs: Optional[dict[str, dict[str, float]]] = None

    def missing_fields(self) -> list[str]:
        """Return high-level list of missing top-level fields."""
        missing = []
        if not self.supply:
            missing.append("supply")
        if not self.demand:
            missing.append("demand")
        if not self.costs:
            missing.append("costs")
        return missing

    def missing_routes(self) -> list[tuple[str, str]]:
        """Return (warehouse, destination) pairs whose cost is not specified."""
        if not self.supply or not self.demand or not self.costs:
            return []
        missing = []
        for warehouse in self.supply:
            for destination in self.demand:
                if (warehouse not in self.costs or
                        destination not in self.costs.get(warehouse, {})):
                    missing.append((warehouse, destination))
        return missing

    def is_complete(self) -> bool:
        return (
            bool(self.supply)
            and bool(self.demand)
            and bool(self.costs)
            and len(self.missing_routes()) == 0
        )

    def follow_up_questions(self) -> list[str]:
        """Human-readable questions for each gap in the problem description."""
        questions = []

        if not self.supply:
            questions.append(
                "What are the supply quantities for each warehouse or source? "
                "(e.g. 'Warehouse A has 1000 units, Warehouse B has 4000 units')"
            )
        if not self.demand:
            questions.append(
                "What are the demand quantities for each destination or customer? "
                "(e.g. 'Store 1 needs 500 units, Store 2 needs 900 units')"
            )
        if not self.costs and self.supply and self.demand:
            questions.append(
                "What are the shipping costs per unit for each route? "
                "(e.g. 'From Warehouse A to Store 1: $2/unit, to Store 2: $4/unit')"
            )
        elif self.supply and self.demand:
            routes = self.missing_routes()
            if routes:
                pairs = ", ".join(f"{w}→{d}" for w, d in routes[:5])
                suffix = f" (and {len(routes)-5} more)" if len(routes) > 5 else ""
                questions.append(
                    f"What are the shipping costs for these routes: {pairs}{suffix}?"
                )

        return questions
