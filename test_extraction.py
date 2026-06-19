"""
Tests for LLM parameter extraction + Pydantic completeness check.
Covers: complete input, missing costs, partial costs, blending input.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from extractor.llm_extractor import (
    extract_transportation_params,
    extract_blending_params,
)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_transport_result(result):
    print(f"Complete : {result.is_complete}")
    if result.is_complete:
        p = result.params
        print(f"Supply   : {p.supply}")
        print(f"Demand   : {p.demand}")
        for src, dests in (p.costs or {}).items():
            print(f"Costs {src}: {dests}")
    else:
        print(f"Missing  : {result.missing_fields}")
        print("Questions:")
        for q in result.follow_up_questions:
            print(f"  ? {q}")


def print_blending_result(result):
    print(f"Complete    : {result.is_complete}")
    if result.is_complete:
        p = result.params
        print(f"Ingredients : {p.ingredients}")
        print(f"Costs       : {p.costs}")
        print(f"Total       : {p.total}")
        print(f"Nutrients   : {p.nutrients}")
        print(f"Requirements: {p.requirements}")
    else:
        print(f"Missing  : {result.missing_fields}")
        print("Questions:")
        for q in result.follow_up_questions:
            print(f"  ? {q}")


# ---------------------------------------------------------------------------
# Test T1: Complete transportation input (Beer Distribution)
# ---------------------------------------------------------------------------
section("T1 – Complete transportation input")
result = extract_transportation_params("""
We have two warehouses:
- Warehouse A can supply 1000 units
- Warehouse B can supply 4000 units

We need to deliver to five stores:
- Store 1: 500 units
- Store 2: 900 units
- Store 3: 1800 units
- Store 4: 200 units
- Store 5: 700 units

Shipping costs per unit:
From A: to Store 1 = $2, to Store 2 = $4, to Store 3 = $5, to Store 4 = $2, to Store 5 = $1
From B: to Store 1 = $3, to Store 2 = $1, to Store 3 = $3, to Store 4 = $2, to Store 5 = $3
""")
print_transport_result(result)

# ---------------------------------------------------------------------------
# Test T2: Missing costs entirely
# ---------------------------------------------------------------------------
section("T2 – Missing costs (should ask for them)")
result = extract_transportation_params("""
Warehouse Alpha has 1000 units available.
Warehouse Beta has 4000 units available.
Customer X needs 500 units.
Customer Y needs 900 units.
Customer Z needs 1800 units.
""")
print_transport_result(result)

# ---------------------------------------------------------------------------
# Test T3: Partial costs (some routes missing)
# ---------------------------------------------------------------------------
section("T3 – Partial costs (some routes missing)")
result = extract_transportation_params("""
Factory North produces 200 items per day.
Factory South produces 300 items per day.
Client A needs 150 items. Client B needs 200 items.
Shipping cost from North to Client A is $5 per item.
Shipping cost from South to Client B is $3 per item.
We don't yet know the cross-route costs.
""")
print_transport_result(result)

# ---------------------------------------------------------------------------
# Test T4: Very vague description (missing almost everything)
# ---------------------------------------------------------------------------
section("T4 – Vague description (supply mentioned, rest missing)")
result = extract_transportation_params("""
We have a warehouse in Kyiv with 500 items that need to go to stores.
""")
print_transport_result(result)

# ---------------------------------------------------------------------------
# Test B1: Blending – Whiskas complete input
# ---------------------------------------------------------------------------
section("B1 – Complete blending input (Whiskas)")
result = extract_blending_params("""
We want to make 100g of cat food.
Ingredients: beef and gel.
Beef costs $0.013 per gram; gel costs $0.008 per gram.

Nutritional content per gram:
- Beef: 0.60 protein, 0.20 fat, 0.02 fibre, 0.01 salt
- Gel:  0.00 protein, 0.00 fat, 0.00 fibre, 0.00 salt

Requirements per 100g can:
- protein  >= 8.0 g
- fat      >= 6.0 g
- fibre    <= 2.0 g
- salt     <= 0.4 g
""")
print_blending_result(result)

# ---------------------------------------------------------------------------
# Test B2: Blending – missing nutritional data
# ---------------------------------------------------------------------------
section("B2 – Blending with missing nutritional data")
result = extract_blending_params("""
We need to blend chicken and rice for a pet food recipe.
Chicken costs $0.02/g, rice costs $0.005/g.
The final product should weigh 200g.
""")
print_blending_result(result)
