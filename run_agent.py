"""
Interactive CLI for the Optimization Agent.
Run:  python3 run_agent.py

The graph uses LangGraph interrupt() for human-in-the-loop steps.
This loop drives it: start → resume with user answer → repeat until END.
"""

import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import compile_graph

DIVIDER = "─" * 60


def print_agent(text: str):
    print(f"\n  {text}\n")


def ask_user(prompt: str = "") -> str:
    if prompt:
        print(f"\n{prompt}\n")
    try:
        return input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        sys.exit(0)


def run():
    print(DIVIDER)
    print("  Optimization Agent  (type 'quit' to exit)")
    print(DIVIDER)

    graph   = compile_graph()
    thread  = {"configurable": {"thread_id": str(uuid.uuid4())}}
    started = False

    while True:
        if not started:
            user_input = ask_user(
                "Describe your optimisation problem "
                "(transportation or blending):"
            )
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            initial_state = {
                "messages": [HumanMessage(content=user_input)]
            }
            events = graph.stream(initial_state, thread, stream_mode="values")
            started = True
        else:
            # Resume after an interrupt with the user's answer
            events = graph.stream(
                Command(resume=user_input),
                thread,
                stream_mode="values",
            )

        # Consume the stream until the next interrupt or END
        last_state = None
        interrupted = False

        for state in events:
            last_state = state

        if last_state is None:
            print_agent("Something went wrong — no state returned.")
            break

        # Check if graph ended or was interrupted
        snapshot = graph.get_state(thread)

        if snapshot.next:
            # Graph is paused at an interrupt node — surface the question
            interrupt_data = snapshot.tasks[0].interrupts[0].value if snapshot.tasks else {}
            question = interrupt_data.get("question", "Please respond:") if isinstance(interrupt_data, dict) else str(interrupt_data)
            user_input = ask_user(question)
            if user_input.lower() in {"quit", "exit", "q"}:
                break
            interrupted = True
        else:
            # Graph finished — print the final explanation
            explanation = last_state.get("explanation", "")
            if explanation:
                print(DIVIDER)
                print_agent(explanation)
                print(DIVIDER)

            print("\nStart a new problem? (Enter to continue, 'quit' to exit)")
            choice = input().strip().lower()
            if choice in {"quit", "exit", "q", "no", "n"}:
                break

            # Reset for a new problem
            thread  = {"configurable": {"thread_id": str(uuid.uuid4())}}
            started = False

        if not interrupted and not started:
            continue


if __name__ == "__main__":
    run()
