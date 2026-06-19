import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()

from agent.graph import compile_graph

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Optimization Agent", page_icon="🔧", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("About")
    st.write(
        "This agent solves **transportation** and **blending** optimization problems. "
        "Describe your problem in plain language and the agent will classify it, "
        "extract all parameters, confirm the model with you, then solve it using "
        "linear programming and explain the results."
    )
    st.markdown("---")
    st.subheader("Sample prompts")
    st.markdown(
        "**Transportation:**\n"
        "> I have warehouses A (300 units) and B (500 units). "
        "Customers 1, 2, 3 need 200, 300, and 300 units. "
        "Shipping costs per unit: A→1 \\$2, A→2 \\$3, A→3 \\$1, "
        "B→1 \\$5, B→2 \\$4, B→3 \\$8. Minimize total cost."
    )
    st.markdown(
        "**Blending:**\n"
        "> Make 100 g of cat food using beef (\\$0.008/g) and gel (\\$0.002/g). "
        "Protein ≥ 8 g, fat ≤ 3 g. Beef has 60% protein and 20% fat; "
        "gel has 0% protein and 0% fat."
    )

# ── Session state initialisation ─────────────────────────────────────────────

def _init_session():
    if "graph" not in st.session_state:
        st.session_state.graph = compile_graph()
    if "thread" not in st.session_state:
        st.session_state.thread = {"configurable": {"thread_id": str(uuid4())}}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "app_state" not in st.session_state:
        st.session_state.app_state = "idle"
    if "interrupt_data" not in st.session_state:
        st.session_state.interrupt_data = None
    if "last_solution" not in st.session_state:
        st.session_state.last_solution = None
    if "last_problem_type" not in st.session_state:
        st.session_state.last_problem_type = None
    if "last_params" not in st.session_state:
        st.session_state.last_params = None


_init_session()

# ── Graph driver ──────────────────────────────────────────────────────────────

def _run_graph(user_input: str, is_resume: bool = False):
    """Drive the graph until it pauses or finishes. Returns last_state."""
    graph = st.session_state.graph
    thread = st.session_state.thread

    last_state = None
    if is_resume:
        payload = Command(resume=user_input)
    else:
        payload = {"messages": [HumanMessage(content=user_input)]}

    for state in graph.stream(payload, thread, stream_mode="values"):
        last_state = state

    return last_state


def _handle_after_run(last_state):
    """Inspect graph snapshot and update session state accordingly."""
    graph = st.session_state.graph
    thread = st.session_state.thread

    if last_state and last_state.get("error"):
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"Error: {last_state['error']}"}
        )
        st.session_state.app_state = "idle"
        return

    snapshot = graph.get_state(thread)

    if snapshot.next:
        # Graph is paused at an interrupt
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        st.session_state.interrupt_data = interrupt_value
        # "approve" interrupt has its own st.info + button UI below — skip chat history
        if interrupt_value.get("node") != "approve":
            question = interrupt_value.get("question", "Please provide more information.")
            st.session_state.chat_history.append({"role": "assistant", "content": question})
        st.session_state.app_state = "interrupted"
    else:
        # Graph finished
        explanation = ""
        if last_state:
            explanation = last_state.get("explanation") or ""
            solution = last_state.get("solution")
            if solution:
                st.session_state.last_solution = solution
                st.session_state.last_problem_type = last_state.get("problem_type")
                st.session_state.last_params = last_state.get("extracted_params")

        if explanation:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": explanation}
            )
        st.session_state.app_state = "done"
        st.session_state.interrupt_data = None

# ── Main title ────────────────────────────────────────────────────────────────

st.title("Optimization Agent")

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Approve interrupt UI ──────────────────────────────────────────────────────

if (
    st.session_state.app_state == "interrupted"
    and st.session_state.interrupt_data
    and st.session_state.interrupt_data.get("node") == "approve"
):
    # Read model_summary directly from graph state — reliable regardless of interrupt format
    graph_state_values = st.session_state.graph.get_state(st.session_state.thread).values
    summary_text = (
        graph_state_values.get("model_summary")
        or st.session_state.interrupt_data.get("summary")
        or ""
    )
    st.markdown("**Review the optimization model before solving:**")
    st.info(summary_text)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Approve & Solve", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "yes"})
            last_state = _run_graph("yes", is_resume=True)
            _handle_after_run(last_state)
            st.rerun()
    with col2:
        st.caption("Or type corrections in the chat input below.")

# ── Solution display ──────────────────────────────────────────────────────────

if (
    st.session_state.app_state == "done"
    and st.session_state.last_solution
    and st.session_state.last_solution.get("status") == "Optimal"
):
    solution = st.session_state.last_solution
    problem_type = st.session_state.last_problem_type
    params = st.session_state.last_params or {}

    with st.expander("Solution Details", expanded=True):
        if problem_type == "transportation":
            total_cost = solution.get("total_cost", 0.0)
            st.metric("Total Shipping Cost", f"${total_cost:,.2f}")

            shipments = solution.get("shipments", {})
            costs_matrix = params.get("costs", {})
            rows = []
            for src, destinations in shipments.items():
                for dst, units in destinations.items():
                    if units and units > 0:
                        unit_cost = (costs_matrix.get(src) or {}).get(dst, 0.0)
                        rows.append({
                            "From": src,
                            "To": dst,
                            "Units": units,
                            "$/unit": unit_cost,
                            "Subtotal $": units * unit_cost,
                        })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

            insights = solution.get("insights", [])
            if insights:
                st.subheader("Key Insights")
                for insight in insights:
                    st.write(f"- {insight}")

        elif problem_type == "blending":
            total_cost = solution.get("total_cost") or 0.0
            total_amount = params.get("total", 100.0) or 100.0
            st.metric("Total Cost", f"${total_cost:.4f}")
            st.metric("Cost per unit", f"${total_cost / total_amount:.6f}")

            mix = solution.get("mix", {})
            mix_pct = solution.get("mix_pct", {})
            costs_map = params.get("costs", {})
            rows = []
            for ingredient, amount in mix.items():
                unit_cost = costs_map.get(ingredient, 0.0)
                rows.append({
                    "Ingredient": ingredient,
                    "Amount": amount,
                    "%": mix_pct.get(ingredient, 0.0),
                    "$/unit": unit_cost,
                    "Subtotal $": amount * unit_cost,
                })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

            if mix_pct:
                st.bar_chart(pd.Series(mix_pct, name="% of blend"))

            insights = solution.get("insights", [])
            if insights:
                st.subheader("Key Insights")
                for insight in insights:
                    st.write(f"- {insight}")

# ── New problem button ────────────────────────────────────────────────────────

if st.session_state.app_state == "done":
    if st.button("Start new problem"):
        st.session_state.graph = compile_graph()
        st.session_state.thread = {"configurable": {"thread_id": str(uuid4())}}
        st.session_state.chat_history = []
        st.session_state.app_state = "idle"
        st.session_state.interrupt_data = None
        st.session_state.last_solution = None
        st.session_state.last_problem_type = None
        st.session_state.last_params = None
        st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Describe your problem or answer above...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    is_resume = st.session_state.app_state == "interrupted"
    last_state = _run_graph(user_input, is_resume=is_resume)
    _handle_after_run(last_state)

    st.rerun()
