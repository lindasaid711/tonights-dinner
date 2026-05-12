# **Tonight's Dinner**

## **1\. Problem & Use Case**

Most weeknights, the question "what's for dinner?" gets answered badly. People default to takeout because they don't have a plan, food in the fridge spoils before it gets used, and money gets wasted on both fronts. The decision itself is genuinely hard as there are multiple factors:what's in the kitchen? what's about to expire? what the household will actually eat? dietary needs? how much time and energy anyone has tonight? Most people don't want to solve that puzzle at 6pm.

**Goal:** Build an agent that owns this single recurring decision *what should I cook tonight?* and gets better at it over time as it learns the household.

**Target audience (MVP):** Single-tenant, designed for one household (mine, initially).

## **2\. Scope**

Decide what to cook tonight given current inventory, dietary constraints, household preferences, and time available.

**Future versions (out of scope for MVP):**

* Weekly meal planning  
* Grocery ordering integration  
* Real-time cooking coach mode  
* Multi-tenant support with auth and user isolation  
* Proactive notifications (e.g., "you have salmon expiring tomorrow") which requires scheduler, notification channel, and a different UX

## **3\. Architecture & Design Pattern**

**Topology: Multi-agent with a supervisor**

Each sub-agent has a distinct prompt, toolset, and success criterion.

| Agent | Role |
| ----- | ----- |
| **Orchestrator** | Understands the user's request, delegates to sub-agents, assembles the final answer |
| **Inventory agent** | Knows what's currently in the kitchen and what's expiring soon |
| **Retrieval agent** | Searches the recipe knowledge base for candidates (RAG) |
| **Evaluator agent** | Scores candidates against dietary needs and constraints |
| **Personalization agent** | Applies household preferences, past reactions, and learned habits |

**Design patterns: ReAct \+ Generate-then-Rank**

Two patterns are used at different layers because they suit different kinds of thinking:

* **ReAct (Reason → Act → Observe → repeat)**: used by the **orchestrator**. Flexible step-by-step reasoning where the next step depends on what the last one returned.

* **Generate-then-Rank (deliberative selection)**: used by the **evaluator**. Generates \~5 candidate dinners, scores each against a constraint rubric (time, ingredients on hand, household preferences, nutrition), and returns the winner with reasoning attached. This is a best-of-N sampling pattern — candidates are evaluated as complete proposals, not as partial reasoning steps — which makes it more precise to call it generate-then-rank than Tree of Thoughts.

## **4\. Memory (CoALA)**

| Memory Type | What it stores | Where it lives |
| ----- | ----- | ----- |
| **Working** | Current conversation, today's inventory snapshot, today's constraints (time available, who's eating, energy level) | Agent context window |
| **Episodic** | Log of past cooking events: "Cooked X on date Y, took 35 min, household reaction, substitutions, leftovers" | Vector DB, embedded events retrievable by similarity |
| **Semantic** | Stable household facts: allergies, equipment, skill level, hard dislikes, dietary patterns | YAML, included in every prompt via prompt caching |
| **Procedural** | Learned heuristics that become rules: "Halve the salt." "Tuesday \= under 25 min." "Don't repeat the same protein two nights in a row." | System prompt or rules file, updated periodically by a reflection process |

**The key loop here is the episodic → procedural.** I would like a periodic agent that reviews the episodic memory, identifies patterns, and writes new rules into procedural memory.

## **5\. Data Retrieval (RAG)**

* **Corpus:** \~10,000 recipes from a curated source (Recipe1M+ or Serious Eats / Food52)  
* **Chunking strategy:** Title \+ ingredient list as one chunk (the primary search target); instructions as a separate chunk fetched only after a recipe is selected  
* **Retrieval:** Hybrid; BM25 over ingredient lists (exact matching matters) \+ dense embeddings over title/description (for vibe-based queries like "something cozy")  
* **Re-ranking:** Cross-encoder re-ranker takes top 50 results and re-orders to top 5 using full constraint context

**6\. Knowledge and Context Graphs**

* **Knowledge graph**: Ingredient ontology with edges like `substitutes_for`, `complements`, `contains_allergen`, `member_of_cuisine`, `seasonal_in`. (I will get this from USDA FoodData Central  
* **Context graph**: Temporal trace of household cooking behavior. Starts empty and grows automatically from each cooking event captured by the feedback loop 

## **7\. Tech Stack**

| Layer | Tool | Role |
| ----- | ---- | ---- |
| **Orchestration** | LangGraph 0.6 | StateGraph with conditional routing; LangGraph Platform manages session checkpointing |
| **LLM** | Claude Sonnet (`claude-sonnet-4-6`) via `langchain-anthropic` | Reasoning, tool calling, and vision (receipt/photo parsing) |
| **Recipe source (MVP)** | TheMealDB API | Free, no-key recipe search; substitutes for a full RAG corpus in the MVP |
| **Persistence** | JSON files (`data/`) | User profile, episodic history, and metrics — survive process restarts, human-readable |
| **Tracing & observability** | LangSmith | Auto-traces every ReAct step; used to inspect tool calls and reasoning quality |
| **Recipe corpus (target)** | Recipe1M+ or Serious Eats / Food52 | Full RAG corpus for production; not yet ingested in MVP |
| **Vector DB (target)** | TBD (e.g., Chroma or pgvector) | Episodic memory retrieval and recipe embedding search in production |

## **8\. Evaluation Framework**

### Groundedness & Accuracy

"Grounded" means every recommendation is supported by what TheMealDB actually returned — Claude is not inventing ingredients or silently violating constraints. Four metrics are tracked automatically after each session:

| Metric | What it checks | Method |
| ------ | -------------- | ------ |
| **Constraint adherence** | Did recommendations avoid allergens, dislikes, and repeated proteins? | Rule-based check against `user_profile.json` |
| **Groundedness** | Are all recommended ingredients present in the raw TheMealDB response? | Diff LLM output against API payload |
| **Instruction following** | Did the agent respect the stated time limit and diversity rules? | Check against session inputs in `AgentState` |
| **Efficiency** | How many tool calls and LLM calls did it take to reach a recommendation? | LangSmith trace metadata |

User thumbs up/down captured per session is the implicit accuracy signal already built into the MVP. The gap is automated scoring that runs without user input.

### Offline Replays

LangSmith captures full traces for every session. Offline replay works as follows:

1. **Build a dataset** of representative sessions in LangSmith — covering edge cases: near-empty fridge, severe allergies, very short time window, all-thumbs-down retry.
2. **Run an LLM-as-judge scorer** that takes a session input + recommendation output and returns scores (1–5) on correctness, completeness, instruction following, and efficiency.
3. **Replay alternative routes** — swap in a different search query or filter order for the same session inputs and compare scores to identify which agent path performs best.

In practice for the MVP, this is a single Python scoring script that replays saved traces through the judge prompt and prints a report. This catches prompt regressions before they reach a live session.

---

## **9\. Error Handling & Robustness**

### Graceful Degradation

The system has three realistic failure points. Each has a defined fallback so the user always gets a response:

| Failure | What breaks | Fallback |
| ------- | ----------- | -------- |
| **TheMealDB API timeout** | No recipe candidates retrieved | Fall back to the last 10 recipes in `episodic_memory.json` as candidates |
| **Claude API timeout** | LLM call fails mid-session | Retry 3× with exponential backoff; if still failing, return a friendly message with cached fallback recipes |
| **`user_profile.json` missing or corrupt** | No constraints loaded | Re-trigger onboarding rather than crashing or recommending with no constraints |
| **Full network failure** | Both API and LLM unavailable | Surface the most recent cached recommendations with an explicit "offline mode" notice |

### Context Poisoning Mitigation

Context poisoning occurs when retrieved content is wrong, irrelevant, or adversarial, and the model treats it as ground truth. The risk is low with TheMealDB (curated API), but becomes real once the system moves to a scraped corpus like Recipe1M+ where data quality is uncontrolled — a recipe might claim to be nut-free but list peanut oil buried in the instructions.

Two mitigations:

1. **Redundancy in retrieval**: Retrieve 10+ candidates and filter to 5 before ranking, so no single bad result dominates the context.

2. **Explicit anomaly validation in the agent prompt**: Before finalizing any recommendation, the agent is instructed to cross-check each recipe against the household allergy list and flag discrepancies:

   > *"Before presenting these recipes, review each one against the household allergy list. Flag any ingredient that appears in the cooking instructions but not the ingredient list, or any item that could contain a hidden allergen. Do not recommend a recipe you cannot confirm is safe."*

   This is a prompt-level change to `nodes.py` that directly addresses the rubric's requirement to prompt the reasoning agent to evaluate retrieved chunks for anomalies before generating a final consensus.

---

## **10\. Why Start with a Single ReAct Agent**

The architecture in Section 3 describes the right long-term design. For the MVP, I collapsed it into a single ReAct agent — and that was a deliberate choice, not a shortcut.

For one household running one session at a time, the coordination overhead of four or more specialized agents adds complexity without adding capability. An orchestrator delegating to an inventory agent, a retrieval agent, an evaluator, and a personalization agent only pays off when those agents are doing genuinely independent work in parallel, or when the system is complex enough that a single agent's context window becomes a constraint. Neither is true here yet.

A single ReAct agent with well-scoped tools — recipe search, constraint filtering, receipt parsing — achieves the same outcome with less failure surface, easier debugging, and faster iteration. The reasoning that would live in four separate agents instead lives in one agent's thought steps, which are fully visible in LangSmith traces.

Multi-agent becomes warranted when the system grows to:
* **Concurrent users** — where per-user agents need to run in parallel with isolated state
* **Truly independent parallel workstreams** — e.g., an inventory-tracking agent running on a daily schedule, separate from the on-demand recipe recommendation session
* **Specialized retrieval at scale** — where a dedicated RAG agent over a 10k+ recipe corpus is too slow to run inline and needs to pre-compute candidate sets asynchronously

The MVP validates the core UX loop and the memory architecture first. The multi-agent topology in Section 3 is the upgrade path once those workstreams actually diverge.

---

## **11\. Demo Build Plan**

The core agent logic is complete. This section documents the four-phase plan to turn it into a demo-ready product.

### Phase 1: Streamlit Chat UI — `app.py` (new file)

Replaces LangGraph Studio with a self-contained `streamlit run app.py` interface. Three components:

**Sidebar** — reads live from disk on every interaction:
- Allergies, dislikes, top 3 foods, learned procedural rules
- Sessions logged and thumbs-up rate from `metrics.json`
- Procedural rules section starts empty and populates visibly after session 5

**Main panel** — `st.chat_message` chat interface with an expandable "Agent is thinking..." section showing each tool call (query sent, results returned) so the class can see the ReAct loop live.

**State machine** in `st.session_state` replaces the LangGraph `interrupt()` flow. The four interrupt-based nodes (`ask_allergies`, `ask_dislikes`, `ask_preferences`, `ask_inventory`, `collect_feedback`) cannot be called directly in Streamlit — `interrupt()` only works inside the LangGraph Platform. Their logic is inlined into a state machine:

```
onboarding_allergies → onboarding_dislikes → onboarding_preferences
→ onboarding_inventory → recommending (auto-triggers) → collecting_feedback → done
```

All other node functions (`load_context`, `react_agent`, `save_feedback`, `reflect`) are called directly from `src/nodes.py`. The `recommending` stage fires automatically on rerun with no user input, calls `react_agent`, then advances to feedback collection.

### Phase 2: Error Handling — `src/tools.py` + `src/nodes.py` (modify)

Three targeted changes, no redesign of core logic:

1. **TheMealDB fallback** in `tools.py`: when the API returns empty, fall back to the last 10 recipes from `episodic_memory.json`. A new `_fallback_from_episodic()` helper reads episodic history and returns prior recipes as candidates so the ReAct loop always has something to work with.

2. **Claude API retry** in `nodes.py`: a `_invoke_with_retry(llm, messages, max_attempts=3)` helper wraps the LLM call with exponential backoff (1s, 2s before 3rd attempt). Transient API errors do not abort the session.

3. **Allergen anomaly validation** added to `_build_system_prompt` in `nodes.py`: a new paragraph instructs the agent to cross-check every recommended recipe's full ingredient list against the household allergy list before outputting — including hidden allergens (e.g., soy sauce contains gluten). Prompt-level change only.

### Phase 3: Evaluation Script — `eval.py` (new file)

Standalone script, run separately, outputs a text report. Four scorers:

| Scorer | What it measures | Method |
| ------ | ---------------- | ------ |
| Constraint adherence | Did any recommended ingredient match the allergy or dislike list? | Rule-based token match against `user_profile.json` |
| Efficiency | Tool calls per session | Stored in episodic entries going forward |
| User satisfaction | Thumbs-up rate per session | Aggregated from `feedback` field in `episodic_memory.json` |
| LLM-as-judge | Completeness, instruction following, satisfaction alignment (1–5 each) | Claude scores last 3 sessions against the household profile |

### Phase 4: Demo Seed Script — `seed_demo.py` (new file)

Pre-populates `episodic_memory.json` with 4 realistic past sessions (varied proteins, mixed feedback, dates spread over past 2 weeks). Sets `metrics.sessions = 4` so the live demo session becomes session 5 and fires `reflect`.

**Effect on demo day:**
- Sidebar shows history and thumbs-up rate immediately (not empty)
- Protein diversity rules engage from the first live session
- After live feedback is submitted, `reflect` fires and procedural rules appear in the sidebar in real time

Script is idempotent: running it twice is safe. A `--force` flag is required to overwrite existing data, preventing accidental overwrites on demo day.

### Demo Day Sequence

```bash
python seed_demo.py        # once, the day before
streamlit run app.py       # no other processes needed
```

1. Sidebar shows 4 sessions and thumbs-up rate from seed data
2. Onboarding is skipped (profile already saved) — goes straight to inventory question
3. Agent runs, tool calls visible in expander
4. Submit feedback → `reflect` fires → procedural rules appear live in sidebar
