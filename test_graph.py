"""
Automated tests for the LangGraph pipeline (Phase 4).
Simulates human responses programmatically so no stdin is needed.

Scenarios:
  G1 — Transportation complete in one shot (no follow-ups)
  G2 — Transportation with missing costs → follow-up loop → solve
  G3 — Blending complete in one shot
  G4 — Approval rejected → correction → re-extract → approve → solve
  G5 — Unknown problem type → graceful exit
"""

import sys, os, uuid
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import compile_graph


def section(title: str):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


def drive_graph(initial_text: str, responses: list[str]) -> dict:
    """
    Run the graph with a scripted list of human responses.
    Returns the final state.
    """
    graph  = compile_graph()
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    resp_iter = iter(responses)

    # Initial invocation
    for state in graph.stream(
        {"messages": [HumanMessage(content=initial_text)]},
        thread,
        stream_mode="values",
    ):
        last_state = state

    while True:
        snapshot = graph.get_state(thread)
        if not snapshot.next:
            break   # graph finished

        # Consume the next scripted response
        try:
            answer = next(resp_iter)
        except StopIteration:
            print("  ⚠  Ran out of scripted responses — graph still waiting")
            break

        interrupt_data = snapshot.tasks[0].interrupts[0].value if snapshot.tasks else {}
        question = interrupt_data.get("question", "") if isinstance(interrupt_data, dict) else ""
        print(f"  [interrupt] {question[:120]}...")
        print(f"  [user]      {answer}")

        for state in graph.stream(
            Command(resume=answer),
            thread,
            stream_mode="values",
        ):
            last_state = state

    return last_state


# ──────────────────────────────────────────────────────────────────────────────
# G1 — Transportation: complete input, approve in one shot
# ──────────────────────────────────────────────────────────────────────────────
section("G1 — Transportation complete (Beer Distribution)")

state = drive_graph(
    initial_text="""
    We have two warehouses: A with 1000 units and B with 4000 units.
    We need to supply stores: 1 needs 500, 2 needs 900, 3 needs 1800, 4 needs 200, 5 needs 700.
    Shipping costs per unit:
    From A: to 1=$2, to 2=$4, to 3=$5, to 4=$2, to 5=$1
    From B: to 1=$3, to 2=$1, to 3=$3, to 4=$2, to 5=$3
    """,
    responses=["yes"],   # approve the model immediately
)

sol = state.get("solution") or {}
ok_cost   = sol.get("total_cost") is not None and abs(sol["total_cost"] - 8600) < 1
ok_status = str(sol.get("status")) == "Optimal"
print(f"\n  Status   : {sol.get('status')}  {'PASS' if ok_status else 'FAIL'}")
print(f"  Cost     : {sol.get('total_cost')}  {'PASS' if ok_cost else 'FAIL'} (expected 8600)")
print(f"  Explain  : {state.get('explanation','')[:120]}...")


# ──────────────────────────────────────────────────────────────────────────────
# G2 — Transportation: missing costs → follow-up loop
# ──────────────────────────────────────────────────────────────────────────────
section("G2 — Transportation with missing costs (follow-up loop)")

state = drive_graph(
    initial_text="""
    Warehouse Alpha has 1000 units. Warehouse Beta has 4000 units.
    Store X needs 2000 units. Store Y needs 3000 units.
    """,
    responses=[
        # Answer to follow-up about costs
        "Shipping costs: Alpha to X = $3/unit, Alpha to Y = $5/unit, "
        "Beta to X = $4/unit, Beta to Y = $2/unit",
        # Approve the reconstructed model
        "yes",
    ],
)

sol = state.get("solution") or {}
ok_status = str(sol.get("status")) == "Optimal"
print(f"\n  Status : {sol.get('status')}  {'PASS' if ok_status else 'FAIL'}")
print(f"  Cost   : {sol.get('total_cost')}")
print(f"  Explain: {state.get('explanation','')[:120]}...")


# ──────────────────────────────────────────────────────────────────────────────
# G3 — Blending complete in one shot (Whiskas 6-ingredient)
# ──────────────────────────────────────────────────────────────────────────────
section("G3 — Blending complete (Whiskas 6-ingredient)")

state = drive_graph(
    initial_text="""
    Make 100g of cat food. Ingredients: Chicken, Beef, Mutton, Rice, WheatBran, Gel.
    Costs: Chicken $0.013, Beef $0.008, Mutton $0.010, Rice $0.002, WheatBran $0.005, Gel $0.001 per gram.
    Protein content: Chicken 10%, Beef 20%, Mutton 15%, Rice 0%, WheatBran 4%, Gel 0%.
    Fat content:     Chicken 8%,  Beef 10%, Mutton 11%, Rice 1%, WheatBran 1%,  Gel 0%.
    Fibre content:   Chicken 0.1%,Beef 0.5%,Mutton 0.3%,Rice 10%,WheatBran 15%, Gel 0%.
    Salt content:    Chicken 0.2%,Beef 0.5%,Mutton 0.7%,Rice 0.2%,WheatBran 0.8%,Gel 0%.
    Requirements: protein >= 8g, fat >= 6g, fibre <= 2g, salt <= 0.4g.
    """,
    responses=["yes"],
)

sol = state.get("solution") or {}
ok_status = str(sol.get("status")) == "Optimal"
ok_cost   = sol.get("total_cost") is not None and abs(sol["total_cost"] - 0.52) < 0.01
beef_pct  = (sol.get("mix_pct") or {}).get("Beef", 0)
ok_beef   = abs(beef_pct - 60.0) < 1.0
print(f"\n  Status  : {sol.get('status')}  {'PASS' if ok_status else 'FAIL'}")
print(f"  Cost    : {sol.get('total_cost')}  {'PASS' if ok_cost else 'FAIL'} (expected ~$0.52)")
print(f"  Beef%   : {beef_pct}  {'PASS' if ok_beef else 'FAIL'} (expected ~60%)")


# ──────────────────────────────────────────────────────────────────────────────
# G4 — Approval rejected → correction → re-approve → solve
# ──────────────────────────────────────────────────────────────────────────────
section("G4 — Edit loop: user rejects model and corrects a cost")

state = drive_graph(
    initial_text="""
    Two plants: P1 with 300 units, P2 with 200 units.
    Two customers: C1 needs 200 units, C2 needs 300 units.
    Shipping: P1 to C1 = $1, P1 to C2 = $3, P2 to C1 = $2, P2 to C2 = $1.
    """,
    responses=[
        # First approval attempt — user spots a typo and corrects
        "Actually the cost from P1 to C2 should be $4, not $3.",
        # Second approval — now correct
        "yes",
    ],
)

sol = state.get("solution") or {}
ok_status = str(sol.get("status")) == "Optimal"
print(f"\n  Status : {sol.get('status')}  {'PASS' if ok_status else 'FAIL'}")
print(f"  Cost   : {sol.get('total_cost')}")
print(f"  Explain: {state.get('explanation','')[:120]}...")


# ──────────────────────────────────────────────────────────────────────────────
# G5 — Unknown problem type
# ──────────────────────────────────────────────────────────────────────────────
section("G5 — Unknown problem type (graceful exit)")

state = drive_graph(
    initial_text="What is the best programming language to learn in 2025?",
    responses=[],   # no interrupts expected
)

expl = state.get("explanation", "")
ok = "couldn't recognise" in expl or "transportation" in expl
print(f"\n  Explanation: {expl}")
print(f"  {'PASS' if ok else 'FAIL'}  graceful unknown response")
