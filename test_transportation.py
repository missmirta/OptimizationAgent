"""
Tests for transportation solver.
(For comparing with predicted results.)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from solvers.transportation import (
    TransportationParams, solve_transportation, format_result
)


def run_test(name: str, params: TransportationParams, expected_cost: float = None):
    print(f"\n{'='*55}")
    print(f"TEST: {name}")
    print('='*55)
    result = solve_transportation(params)
    print(format_result(params, result))
    if expected_cost is not None and result.total_cost is not None:
        match = abs(result.total_cost - expected_cost) < 0.01
        print(f"Predicted cost: ${expected_cost:.2f} → {'PASS' if match else 'FAIL'}")


# ---------------------------------------------------------------------------
# Test 1: Beer Distribution Problem
# ---------------------------------------------------------------------------
run_test(
    name="Beer Distribution",
    params=TransportationParams(
        supply={"A": 1000, "B": 4000},
        demand={"1": 500, "2": 900, "3": 1800, "4": 200, "5": 700},
        costs={
            "A": {"1": 2, "2": 4, "3": 5, "4": 2, "5": 1},
            "B": {"1": 3, "2": 1, "3": 3, "4": 2, "5": 3},
        }
    ),
    expected_cost=8600.0
)

# ---------------------------------------------------------------------------
# Test 2: Over supply → automatic dummy destination
# supply=5000, demand=4100 → 900 extra should be absorbed
# ---------------------------------------------------------------------------
run_test(
    name="Over supply (auto dummy destination)",
    params=TransportationParams(
        supply={"A": 1000, "B": 4000},
        demand={"1": 500, "2": 900, "3": 1800, "4": 200, "5": 700},
        costs={
            "A": {"1": 2, "2": 4, "3": 5, "4": 2, "5": 1},
            "B": {"1": 3, "2": 1, "3": 3, "4": 2, "5": 3},
        }
    )
)

# ---------------------------------------------------------------------------
# Test 3: Minimal example (2×2)
# ---------------------------------------------------------------------------
run_test(
    name="Minimal example (2x2)",
    params=TransportationParams(
        supply={"W1": 100, "W2": 100},
        demand={"B1": 100, "B2": 100},
        costs={
            "W1": {"B1": 1, "B2": 10},
            "W2": {"B1": 10, "B2": 1},
        }
    ),
    expected_cost=200.0
)

# ---------------------------------------------------------------------------
# Test 4: Over demand → dummy source added automatically
# ---------------------------------------------------------------------------
run_test(
    name="Over demand (auto dummy source)",
    params=TransportationParams(
        supply={"A": 1000},
        demand={"1": 600, "2": 600},
        costs={"A": {"1": 1, "2": 2}}
    )
)
