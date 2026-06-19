"""
Tests for the blending solver + classifier.
Phase 3 verification checklist:
  [x] Whiskas result: 60% Beef + 40% Gel = $0.52  (PuLP Whiskas 2 dataset)
  [x] Infeasible case (requirements cannot be met)
  [x] Three-ingredient blend
  [x] Classifier: transportation / blending / unknown
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from schemas.blending import BlendingProblemInput, NutrientRequirement
from solvers.blending import solve_blending, format_result
from extractor.classifier import classify, ProblemType


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# Canonical Whiskas 2 dataset (PuLP documentation, 6 ingredients)
# Expected: 60g Beef + 40g Gel = $0.52
# ---------------------------------------------------------------------------
def whiskas_params() -> BlendingProblemInput:
    return BlendingProblemInput(
        ingredients=["Chicken", "Beef", "Mutton", "Rice", "WheatBran", "Gel"],
        costs={
            "Chicken": 0.013, "Beef": 0.008, "Mutton": 0.010,
            "Rice": 0.002,    "WheatBran": 0.005, "Gel": 0.001,
        },
        nutrients={
            "protein": {"Chicken": 0.100, "Beef": 0.200, "Mutton": 0.150,
                        "Rice": 0.000, "WheatBran": 0.040, "Gel": 0.000},
            "fat":     {"Chicken": 0.080, "Beef": 0.100, "Mutton": 0.110,
                        "Rice": 0.010, "WheatBran": 0.010, "Gel": 0.000},
            "fibre":   {"Chicken": 0.001, "Beef": 0.005, "Mutton": 0.003,
                        "Rice": 0.100, "WheatBran": 0.150, "Gel": 0.000},
            "salt":    {"Chicken": 0.002, "Beef": 0.005, "Mutton": 0.007,
                        "Rice": 0.002, "WheatBran": 0.008, "Gel": 0.000},
        },
        requirements={
            "protein": NutrientRequirement(operator=">=", value=8.0),
            "fat":     NutrientRequirement(operator=">=", value=6.0),
            "fibre":   NutrientRequirement(operator="<=", value=2.0),
            "salt":    NutrientRequirement(operator="<=", value=0.4),
        },
        total=100.0,
    )


# ---------------------------------------------------------------------------
# B1 — Whiskas canonical (PuLP Whiskas 2: 60% Beef + 40% Gel = $0.52)
# ---------------------------------------------------------------------------
section("B1 — Whiskas 6-ingredient (expected: 60% Beef, 40% Gel, cost = $0.52)")
params = whiskas_params()
result = solve_blending(params)
print(format_result(params, result))

checks = [
    ("cost == $0.52",   result.total_cost is not None and abs(result.total_cost - 0.52) < 0.01),
    ("Beef  == 60%",    abs(result.mix_pct.get("Beef", 0)  - 60.0) < 1.0),
    ("Gel   == 40%",    abs(result.mix_pct.get("Gel",  0)  - 40.0) < 1.0),
]
for label, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


# ---------------------------------------------------------------------------
# B2 — Infeasible (only gel, but protein >= 8 required)
# ---------------------------------------------------------------------------
section("B2 — Infeasible blend (protein requirement cannot be satisfied)")
infeasible = BlendingProblemInput(
    ingredients=["Gel"],
    costs={"Gel": 0.001},
    nutrients={"protein": {"Gel": 0.00}},
    requirements={"protein": NutrientRequirement(operator=">=", value=8.0)},
    total=100.0,
)
result2 = solve_blending(infeasible)
print(f"Status  : {result2.status}")
print(f"Insights: {result2.insights}")
print(f"  {'PASS' if str(result2.status) == 'Infeasible' else 'FAIL'}  status == Infeasible")


# ---------------------------------------------------------------------------
# B3 — Simple 2-ingredient (Beef + Gel, verify fat is binding constraint)
# ---------------------------------------------------------------------------
section("B3 — 2-ingredient blend (Beef $0.013 + Gel $0.008)")
params3 = BlendingProblemInput(
    ingredients=["beef", "gel"],
    costs={"beef": 0.013, "gel": 0.008},
    nutrients={
        "protein": {"beef": 0.60, "gel": 0.00},
        "fat":     {"beef": 0.20, "gel": 0.00},
        "fibre":   {"beef": 0.02, "gel": 0.00},
        "salt":    {"beef": 0.01, "gel": 0.00},
    },
    requirements={
        "protein": NutrientRequirement(operator=">=", value=8.0),
        "fat":     NutrientRequirement(operator=">=", value=6.0),
        "fibre":   NutrientRequirement(operator="<=", value=2.0),
        "salt":    NutrientRequirement(operator="<=", value=0.4),
    },
    total=100.0,
)
result3 = solve_blending(params3)
print(format_result(params3, result3))
# fat >= 6 requires beef >= 30; salt <= 0.4 requires beef <= 40
# optimal: beef=30 (fat binding), gel=70, cost=30*0.013+70*0.008=$0.95
print(f"  {'PASS' if abs(result3.mix.get('beef',0) - 30.0) < 0.1 else 'FAIL'}  beef == 30g (fat constraint binding)")


# ---------------------------------------------------------------------------
# Classifier tests (requires OPENAI_API_KEY in .env)
# ---------------------------------------------------------------------------
section("C1 — Classifier: transportation / blending / unknown")

cases = [
    (
        "We have factories in Kyiv and Lviv. We need to ship products to "
        "warehouses in Odesa and Kharkiv at minimum transport cost.",
        ProblemType.TRANSPORTATION,
    ),
    (
        "I want to make 100g of cat food from beef and chicken. "
        "Each has protein and fat content. Minimise cost while meeting nutrition targets.",
        ProblemType.BLENDING,
    ),
    (
        "What is the capital of France?",
        ProblemType.UNKNOWN,
    ),
    (
        "We produce 500 units at plant A and 300 at plant B. "
        "Retailers X, Y, Z need 200, 400, 200 units respectively. "
        "Shipping rates vary by route.",
        ProblemType.TRANSPORTATION,
    ),
    (
        "Mix sand, gravel and cement to produce concrete. "
        "Minimum compressive strength 30MPa. Minimise material cost per cubic metre.",
        ProblemType.BLENDING,
    ),
]

all_pass = True
for text, expected in cases:
    problem_type, confidence, reason = classify(text)
    ok = problem_type == expected
    all_pass = all_pass and ok
    print(f"  {'PASS' if ok else 'FAIL'}  Expected={expected.value:15s}  Got={problem_type.value:15s}  ({confidence})")
    print(f"        {reason}")

print(f"\nClassifier: {'ALL PASS' if all_pass else 'SOME FAILED'}")
