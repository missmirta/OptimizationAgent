# Optimization Agent — PoC

An autonomous agent that understands optimization problems described in plain language,
formalizes them as linear programs, solves them with PuLP, and explains the results.

Supports two problem types: **Transportation** (minimum-cost distribution) and
**Blending** (minimum-cost mixture subject to composition constraints).

---

## Key Architectural Decision

> **The LLM extracts parameters. Code does the math.**

The naive approach — ask an LLM to generate PuLP code and `exec()` it — has two critical
flaws: it enables arbitrary code execution, and hallucinations in generated code silently
produce wrong answers.

This project uses a strict separation of concerns:

| Layer | Nature | Responsibility |
|---|---|---|
| LLM (GPT-4o-mini) | Probabilistic | Classifies problem type; extracts structured parameters from free text; explains results in plain language |
| Solver (PuLP + CBC) | Deterministic | Takes a validated JSON schema, builds the LP model, computes the exact optimum |

The LLM never touches a number. The solver never reads text. Each component does only
what it is best at. This makes the system both safe and predictable — exactly the pattern
worth demonstrating.

---

## Pipeline

```
User message
     │
     ▼
 [Classify]          GPT-4o-mini: transportation | blending | unknown
     │
     ▼
 [Extract params]    GPT-4o-mini tool-calling → Pydantic schema
     │
     ▼
 [Completeness?] ── missing ──► [Ask follow-up] ◄── loops back ──┐
     │ complete                        │                           │
     ▼                                 └───────────────────────────┘
 [Build model summary]   human-readable objective + constraints
     │
     ▼
 [APPROVE?] ── corrections ──► [Extract params]  (edit loop)
     │ yes
     ▼
 [Solve (PuLP)]       deterministic LP solve
     │
     ▼
 [Explain results]    shipments / mix table + binding constraints + insights
```

Two human-in-the-loop cycles, both implemented via LangGraph `interrupt()`:
1. **Follow-up questions** — agent asks for missing data; user answers in chat; extraction re-runs with full context
2. **Model approval** — agent shows the model before solving; user can approve or correct

---

## Project Structure

```
optimizationAgent/
│
├── app.py                   Streamlit chat UI (entry point)
├── run_agent.py             Interactive CLI (alternative entry point)
│
├── agent/
│   ├── state.py             AgentState TypedDict (shared graph state)
│   ├── nodes.py             LangGraph node functions (8 nodes)
│   └── graph.py             Graph wiring + compile_graph()
│
├── extractor/
│   ├── classifier.py        Problem-type classifier (few-shot GPT)
│   └── llm_extractor.py     Parameter extractor (OpenAI tool-calling)
│
├── schemas/
│   ├── transportation.py    Pydantic schema + completeness check
│   └── blending.py          Pydantic schema + completeness check
│
├── solvers/
│   ├── transportation.py    PuLP transportation solver
│   └── blending.py          PuLP blending solver
│
└── test_*.py                Phase verification tests
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Solver | **PuLP + CBC** | Free, covers both canonical PoC cases; Gurobi/CuOpt in future work |
| Agent graph | **LangGraph** | State machine with `interrupt()` for human-in-the-loop cycles |
| LLM | **GPT-4o-mini** | Tool-calling for structured extraction; cheap and fast for a PoC |
| Parameter validation | **Pydantic v2** | Schema completeness check — missing field = follow-up question |
| UI | **Streamlit** | Chat UI in ~250 lines, no frontend code |

---

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url>
cd optimizationAgent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and add:  OPENAI_API_KEY=sk-proj-...

# 4. Launch the UI
streamlit run app.py
# → opens at http://localhost:8501

# 5. (Optional) CLI mode
python run_agent.py
```

### Run the tests

```bash
python test_transportation.py   # Phase 1 — solver, no API needed
python test_blending.py         # Phase 3 — solver + classifier
python test_extraction.py       # Phase 2 — LLM extraction
python test_graph.py            # Phase 4 — full graph with scripted responses
```

---

## Example Dialogues

### 1 — Transportation (complete input, one shot)

> **User:** I have two warehouses: A with 1000 units and B with 4000 units. I need to
> deliver to five stores — 1 needs 500 units, 2 needs 900, 3 needs 1800, 4 needs 200,
> 5 needs 700. Shipping costs per unit: from A: to 1=$2, to 2=$4, to 3=$5, to 4=$2,
> to 5=$1. From B: to 1=$3, to 2=$1, to 3=$3, to 4=$2, to 5=$3.

**Agent:**
> Here is the optimisation model I've built:
>
> Transportation Problem — minimise total shipping cost
>
> Supply: A: 1000 units, B: 4000 units
> Demand: Store 1: 500, Store 2: 900, Store 3: 1800, Store 4: 200, Store 5: 700
> Supply (5000) ≠ Demand (4100) — a dummy node will be added automatically.
>
> Type **yes** to solve it, or describe any corrections.

> **User:** yes

**Agent:**
> Optimal solution — total shipping cost: **$8,600.00**
>
> | From | To | Units | $/unit | Subtotal |
> |---|---|---|---|---|
> | A | 1 | 300 | $2 | $600 |
> | A | 5 | 700 | $1 | $700 |
> | B | 1 | 200 | $3 | $600 |
> | B | 2 | 900 | $1 | $900 |
> | B | 3 | 1800 | $3 | $5,400 |
> | B | 4 | 200 | $2 | $400 |
>
> **Key insights:**
> - Warehouse A is fully exhausted (1000 units)
> - Warehouse B has 900 units left — not needed
> - Most expensive active route: B→1 ($3/unit) — forced by demand constraints
> - Demand met exactly (no surplus) at all 5 stores

---

### 2 — Transportation (missing costs → follow-up loop)

> **User:** I have a warehouse in Kyiv with 500 units and one in Lviv with 300 units.
> Customer in Odesa needs 400 units, customer in Kharkiv needs 400 units.

**Agent:**
> I need a bit more information to set up the model:
> 1. What are the shipping costs per unit for each route?
>    (e.g. "From Kyiv to Odesa: $3/unit, to Kharkiv: $5/unit")

> **User:** Kyiv to Odesa $3, Kyiv to Kharkiv $7, Lviv to Odesa $6, Lviv to Kharkiv $4.

**Agent:**
> Here is the optimisation model I've built:
>
> Transportation Problem — minimise total shipping cost
> Supply (800) ≠ Demand (800) — balanced.
> ...
> Type **yes** to solve it.

> **User:** yes

**Agent:**
> Optimal solution — total cost: **$3,200.00**
>
> - Kyiv → Odesa: 400 units @ $3
> - Lviv → Kharkiv: 300 units @ $4  *(Lviv exhausted; Kharkiv short 100 units from dummy)*
>
> **Insight:** Lviv's cheaper route to Kharkiv ($4) wins over Kyiv's ($7); Odesa served
> entirely from Kyiv despite longer distance because Kyiv→Odesa ($3) beats Lviv→Odesa ($6).

---

### 3 — Blending (Whiskas cat food)

> **User:** I want to make 100g of cat food. I can use beef ($0.008/g) and gel ($0.001/g).
> Beef has 20% protein, 10% fat, 0.5% fibre, 0.5% salt.
> Gel has no nutritional value.
> Requirements per 100g: protein ≥ 8g, fat ≥ 6g, fibre ≤ 2g, salt ≤ 0.4g.

**Agent:**
> Blending Problem — minimise cost per 100 units
>
> Ingredients: beef $0.008/g, gel $0.001/g
> Constraints: protein ≥ 8g, fat ≥ 6g, fibre ≤ 2g, salt ≤ 0.4g
>
> Type **yes** to solve.

> **User:** yes

**Agent:**
> Optimal blend — total cost: **$0.5200** per 100g can
>
> | Ingredient | Amount | % | $/g | Subtotal |
> |---|---|---|---|---|
> | beef | 60.0 g | 60.0% | $0.008 | $0.480 |
> | gel | 40.0 g | 40.0% | $0.001 | $0.040 |
>
> **Key insights:**
> - Dominant ingredient: beef (60%) — fat constraint forces at least 60g (0.10 × 60 = 6g fat)
> - Binding constraint: fat ≥ 6g — this limits how much cheap gel can replace beef
> - Remaining 40g is cheapest filler (gel at $0.001/g)
> - Increasing the fat requirement even slightly would raise cost proportionally

---

## Future Work

### Higher-performance solvers
- **[Gurobi](https://www.gurobi.com/)** — commercial solver, significantly faster for large models; drop-in PuLP backend
- **[NVIDIA CuOpt](https://developer.nvidia.com/cuopt)** — GPU-accelerated routing and supply-chain optimization; ideal for large-scale transportation

### MCP Server
Wrap the solvers as [Model Context Protocol](https://modelcontextprotocol.io/) tools.
Any MCP-capable agent (Claude, Cursor, etc.) could then call `solve_transportation` or
`solve_blending` as a structured tool — demonstrating true agentic tool-use rather than
UI-driven interaction.

```python
# Sketch of an MCP tool definition
@mcp.tool()
def solve_transportation(supply: dict, demand: dict, costs: dict) -> dict:
    params = TransportationParams(supply=supply, demand=demand, costs=costs)
    return asdict(solve_transportation(params))
```

### Additional problem types
| Type | Description |
|---|---|
| **Assignment** | Match n workers to n jobs at minimum cost |
| **Production planning** | Multi-period inventory + production scheduling |
| **Network flow** | Generalised min-cost flow (subsumes transportation) |
| **Knapsack / bin packing** | Resource allocation under capacity constraints |

Each new type requires only: a Pydantic schema, a PuLP solver template, and a few
few-shot examples for the classifier. The LangGraph pipeline is unchanged.

### LLM-powered insights
Currently insights are generated by deterministic heuristics (binding constraints, slack
values). Feeding the solver result back to the LLM for a natural-language explanation
would produce richer, context-aware analysis:
> *"The fat constraint is binding because gel — the cheapest ingredient at $0.001/g —
> contributes zero fat. Every extra gram of fat required forces at least 10g of beef,
> adding $0.08 to the can cost."*

### Sensitivity analysis
PuLP exposes shadow prices and allowable ranges. Surfacing these lets users answer:
*"How much would it save if the shipping cost from B to 2 dropped by $1?"*

---

## MCP Server (Stretch Goal — Implemented)

The solvers are also exposed as an **MCP (Model Context Protocol) server** — the same
deterministic functions, callable by any MCP-capable client without touching the agent code.

```
LangGraph agent  ──import──►  solve_transportation()   (current default)

Claude Desktop   ──MCP──►  mcp_server.py  ──import──►  solve_transportation()
Any MCP client   ──MCP──►      (stdio)                  solve_blending()
```

### Why this matters

This is the true **agentic tool-use** pattern: the solver is a standalone service.
Claude Desktop can call it from a plain chat conversation with zero agent code.
The server has no knowledge of LangGraph; LangGraph has no knowledge of MCP.

### Start the server

```bash
# Stdio transport — for Claude Desktop or any MCP client
python mcp_server.py

# Browser inspector — for manual testing
mcp dev mcp_server.py
```

### Tools exposed

| Tool | Description |
|---|---|
| `solve_transportation_problem` | Minimise shipping cost from sources to destinations |
| `solve_blending_problem` | Minimise blend cost subject to nutritional constraints |

Both return structured JSON — `{"success": true, ...}` or `{"success": false, "error": ..., "hint": ...}`.
The LLM never receives a Python traceback.

### Connect Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "optimization-agent": {
      "command": "python3",
      "args": ["/absolute/path/to/optimizationAgent/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop — the two solver tools appear automatically in the tool panel.

### Verified behaviour

```
initialize  → serverInfo: {name: "optimization-agent"}
tools/list  → [solve_transportation_problem, solve_blending_problem]
tools/call  → solve_transportation_problem(Beer Distribution) → total_cost: 8600.0  ✓
tools/call  → solve_blending_problem(Whiskas 6-ingredient)   → total_cost: 0.52     ✓
tools/call  → infeasible blend                               → {success: false, hint: ...} ✓
```
# OptimizationAgent
