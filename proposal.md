# Tonight's Dinner — Project Proposal
**Course:** Applied AI Agents  
**Date:** May 2026  
**Status:** MVP built and running end-to-end

---

## 1. Problem & Motivation

Most weeknights, the question *"what's for dinner?"* gets answered badly. People default to takeout not because they lack food, but because the decision is genuinely hard at 6pm: What's in the kitchen? What's about to expire? What will everyone actually eat? What are the dietary constraints? How much time and energy is there tonight?

**Goal:** Build an agent that owns this single recurring decision and gets meaningfully better at it over time as it learns the household — reducing decision fatigue, cutting food waste, and saving money.

**Target:** Single-tenant, one household. Not a recipe search engine — a system that *knows you* and improves with every session.

---

## 2. What Is Built (MVP)

A working end-to-end LangGraph agent deployed locally with LangGraph Platform:

1. **One-time onboarding** — the agent asks about allergies, dislikes, and food preferences (top 3 from 20 options)
2. **Daily inventory capture** — the user describes what's in their kitchen, or uploads a grocery receipt photo which Claude parses via vision
3. **ReAct recommendation loop** — the agent searches a recipe API, filters against constraints, enforces nutritional diversity (no repeated proteins vs. the last 7 days), and returns 3 recommendations with protein, key ingredients, and cook time
4. **Lightweight feedback** — thumbs up / down per recipe; if all 3 are rejected, the agent retries with a different search strategy (up to 2 retries)
5. **Persistent memory** — user profile, session history, and evaluation metrics survive across process restarts
6. **Periodic reflection** — every 5 sessions, a reflection node reviews the full feedback history and writes learned rules back into the system prompt (e.g. *"avoid fish dishes — consistently low feedback"*, *"prefer pasta dishes under 30 min"*)

**Tech stack:** LangGraph 0.6 · Claude Sonnet 4.6 (LLM + vision) · TheMealDB (recipe API) · LangSmith (tracing + observability) · JSON files for persistent memory (dev) · Streamlit (chat UI)

---

## 3. Agent Topology: Why Single-Agent for the MVP

### Original design intent
The full design calls for a **multi-agent supervisor architecture** with five specialised sub-agents:

| Agent | Responsibility |
|---|---|
| Orchestrator | Understands the request, delegates, assembles the final answer |
| Inventory agent | Tracks kitchen contents and expiry signals |
| Retrieval agent | Searches the recipe corpus (RAG) |
| Evaluator agent | Scores candidates against constraints using Tree of Thoughts |
| Personalization agent | Applies household preferences, past reactions, and learned habits |

### The principled argument for single-agent at this stage

The decision to implement a single agent for the MVP is not a shortcut — it is the correct engineering call at this point in the system's lifecycle, for four reasons:

**1. The task is a sequential pipeline, not a concurrent network.**
Each step depends strictly on the output of the previous one: you cannot retrieve recipes before knowing the inventory; you cannot evaluate candidates before retrieving them; you cannot personalize before evaluation. There is no meaningful parallelism to exploit with multiple agents at this stage. Introducing a supervisor would add coordination overhead (extra LLM calls, state serialisation, routing logic) without unlocking any parallel execution.

**2. The system is data-starved, not reasoning-starved.**
The bottleneck is not the quality of the reasoning inside each node — it is the absence of enough feedback history to justify specialised agents. The Personalization agent, for example, can only do meaningful work once episodic memory has accumulated enough sessions to identify real patterns. A single agent with a well-structured system prompt captures the same reasoning at a fraction of the cost while the data flywheel builds.

**3. A monolithic agent is dramatically easier to debug and improve in the early feedback loop.**
With LangSmith tracing, every node, tool call, and LLM response is visible in a single trace. If the recommendations are wrong, the root cause is immediately identifiable. In a multi-agent system, a bad recommendation might originate in the retrieval agent, the evaluator, or the personalization layer — tracing that causal chain across agent boundaries is significantly more complex.

**4. Premature decomposition locks in incorrect boundaries.**
The right boundaries between agents should emerge from observed failure modes, not be imposed upfront. After 30 sessions of real use, the data will show whether the retrieval logic is the bottleneck, whether the evaluation needs richer reasoning, or whether the personalization layer needs its own context window. Decomposing now risks drawing the wrong lines.

### The clear trigger for moving to multi-agent

The architecture should decompose into the full supervisor model when **any two of the following are true:**
- The recipe corpus grows beyond ~1,000 recipes and requires dedicated RAG infrastructure (embedding store, re-ranker)
- The evaluation step becomes complex enough to benefit from Tree of Thoughts reasoning in an isolated context window
- The system expands to multi-tenant (multiple households), making the personalization layer stateful and independently deployable
- Session latency becomes a problem and inventory + retrieval can be parallelised

### The honest argument for multi-agent (steel-manning the alternative)

A principled case can be made that multi-agent is already the right call, even at MVP scale:

- **Separation of concerns produces better prompts.** A prompt written for a single agent must serve retrieval, evaluation, and personalization simultaneously — it becomes long and diluted. A retrieval agent with a short, focused prompt likely retrieves better than a general agent with a bloated system prompt.
- **Independent failure and retry.** If retrieval returns zero results, a supervisor can retry just the retrieval agent with a different query without rerunning the full pipeline. In a single agent, failure anywhere requires re-entering the ReAct loop from scratch.
- **The evaluator and retriever genuinely require different reasoning styles.** ReAct (sequential tool-calling) suits retrieval. Tree of Thoughts (generate → score → select) suits evaluation. These patterns sit awkwardly in the same context window and are cleaner as separate agents with separate prompts.
- **Testability.** Each agent can be evaluated independently: precision/recall for the retrieval agent, constraint adherence rate for the evaluator, preference alignment score for the personalization agent. In a single agent, these concerns are entangled.

**Verdict:** For a course submission, either architecture is defensible. The single-agent MVP was chosen because the data flywheel is not yet turning — but the multi-agent design is the explicit next step and the architectural interfaces are already drawn in the design.

### Tradeoff matrix: single vs. multi-agent

| Dimension | Single Agent | Multi-Agent |
|---|---|---|
| **Model size required** | Must use a large capable model for all tasks | Right model per task — cheap fast model for extraction, powerful model for evaluation |
| **Cost per session** | Fewer LLM calls, but each call is expensive (large model, large context) | More LLM calls, but each can be cheaper — net cost depends on task mix |
| **Context window** | Bloats over time — system prompt + all tools + memory all compete for space | Each agent has a short, focused context — prompts stay tight |
| **Prompt quality** | One prompt serves retrieval, evaluation, and personalisation — diluted | Each agent has a single-purpose prompt — focused prompts outperform diluted ones |
| **Parallelism** | None — strictly sequential | Independent tasks (e.g. inventory check + recipe retrieval) can run concurrently |
| **Failure isolation** | One failure touches the whole pipeline | Failure in retrieval only retries retrieval — rest of the pipeline untouched |
| **Debugging** | Single LangSmith trace — root cause immediately visible | Causality spans multiple traces — harder to identify where a bad recommendation originated |
| **Build time** | One prompt, one graph, one set of tools | Each agent needs its own prompt, tools, success criteria, and evaluation |
| **Session latency** | Lower — one model call chain | Higher — 4–5 sequential agent calls, each adding round-trip latency |
| **Testability** | Concerns are entangled — hard to isolate retrieval quality from evaluation quality | Each agent independently measurable (precision/recall for retrieval, constraint adherence for evaluation) |
| **Coordination overhead** | None | Orchestrator itself is an LLM call — routing tokens cost money |
| **State handoff** | State lives in one place | Results must be serialised between agents — context can be lost or distorted in handoff |

### The right-model-per-task argument in numbers

The most compelling case for multi-agent is cost efficiency through model specialisation:

| Agent | Task complexity | Right model | Approx. cost/call |
|---|---|---|---|
| Inventory parser | Pure extraction — read a receipt, list ingredients | Claude Haiku | ~$0.001 |
| Recipe retrieval | Structured search reasoning | Claude Sonnet | ~$0.010 |
| Evaluator | Nuanced multi-constraint scoring (Tree of Thoughts) | Claude Sonnet | ~$0.015 |
| Personalisation | Pattern matching over feedback history | Claude Sonnet | ~$0.008 |
| **Single agent (current)** | All of the above in one call | Claude Sonnet | ~$0.018 |

At low session volume (30 sessions/month), the single agent is marginally cheaper because you avoid orchestrator overhead. At high volume (300+ sessions/month across multiple households), the per-task model optimisation in multi-agent becomes meaningfully cheaper.

### The key question that resolves the tradeoff

> **Are the subtasks different enough in complexity and frequency to justify separate models and the coordination overhead?**

For this app today: **not yet**. All tasks run once per session in sequence, the household is single-tenant, and the recipe corpus is small. The crossover point arrives when:
- The inventory agent runs on a **schedule** (expiry monitoring) — independently of the recommendation flow
- The retrieval agent works over a **large corpus** — expensive enough to justify a specialised, cheaper model
- The system scales to **multiple households** — personalisation becomes stateful and must be isolated per user

Until then, the single agent captures the same reasoning at lower coordination cost, with a far simpler debugging story.

---

## 4. Memory Architecture (CoALA)

The system implements all four CoALA memory tiers, with different levels of sophistication at MVP vs. full design:

| Memory Type | What it stores | MVP implementation | Full design |
|---|---|---|---|
| **Working** | Today's session state: inventory snapshot, current conversation, constraints | LangGraph `AgentState` (context window) — cleared each session | Same — this is correct by design |
| **Episodic** | Log of past sessions: date, recipes shown, proteins, feedback | `data/episodic_memory.json` — flat list of session records | Vector DB (e.g. Pinecone) — embedded cooking events retrievable by similarity (e.g. *"find sessions where the user liked a vegetarian dish on a weeknight"*) |
| **Semantic** | Stable household facts: allergies, dislikes, favourite foods, skill level, equipment | `data/user_profile.json` — injected into every system prompt | YAML file with LangChain prompt caching — zero marginal cost per call once cached |
| **Procedural** | Learned heuristics: *"avoid fish"*, *"prefer under 30 min on weekdays"* | `procedural_rules` field in `user_profile.json`, written by the reflection node | Formalised rules file, versioned, with confidence scores per rule |

### The key loop: episodic → procedural

The most important architectural feature is the feedback loop from episodic to procedural memory. Every 5 sessions, a `reflect` node:
1. Reads the full episodic history
2. Asks Claude to identify patterns across sessions (e.g. *"fish dishes have 0% thumbs-up rate over 6 sessions"*)
3. Writes new rules directly into the procedural memory (system prompt)
4. Those rules take effect immediately in the next session — the system learns without any manual intervention

This closes the loop between observed behaviour and future decision-making. It is a lightweight implementation of the continual learning principle described in the CoALA framework.

---

## 5. Data Retrieval (RAG) — Design and Deferral Rationale

### MVP approach
The MVP uses **TheMealDB** — a free structured recipe API — as a stand-in for a full RAG pipeline. The agent issues keyword queries, receives structured JSON, and filters in-context. This is intentionally not RAG.

### Why RAG was deferred
RAG requires a corpus worth retrieving over. The MVP's primary goal is to validate the feedback loop and preference-learning mechanism. Until the system has demonstrated it can learn household preferences, there is no signal to guide corpus curation or embedding strategy. Investing in RAG infrastructure before the feedback loop is proven would be premature optimisation.

### Full RAG design (Phase 2)

**Corpus:** ~10,000 recipes from Recipe1M+ or Serious Eats / Food52

**Chunking strategy — two-level:**
- *Primary chunk:* title + ingredient list as one unit. This is the search target — most queries are ingredient-driven ("what can I make with chicken and garlic?") or vibe-driven ("something cozy")
- *Secondary chunk:* instructions fetched lazily, only after a recipe is selected. Instructions are long, expensive to embed, and not needed for candidate ranking

**Retrieval — hybrid:**
- BM25 over ingredient lists (exact term matching matters — "shellfish" must match "shellfish")
- Dense embeddings over title + description (semantic matching for vibe queries)
- Score fusion via Reciprocal Rank Fusion

**Re-ranking:**
- Cross-encoder re-ranker takes the top 50 hybrid results and re-orders to top 5 using the full constraint context: allergies, dislikes, nutritional diversity constraint, household preferences, and recent protein history
- The re-ranker is the correct place to apply the richest context — it sees fewer candidates and can afford a deeper comparison

---

## 6. Knowledge and Context Graphs

### Knowledge graph (ingredient ontology)
A structured representation of ingredient relationships, sourced from USDA FoodData Central:

```
chicken  ──substitutes_for──▶  tofu
salmon   ──contains_allergen──▶  fish
garlic   ──complements──▶  olive oil
broccoli ──seasonal_in──▶  autumn, winter
sriracha ──member_of_cuisine──▶  Asian
```

**Role in the system:** the knowledge graph is the bridge between *what the user has* and *what recipes are plausible*. Without it, the retrieval agent can only do string matching. With it, the system can reason: *"the user has tofu and the recipe calls for chicken — this is a valid substitution, include it."*

**MVP status:** not yet implemented. In the MVP, constraint filtering is done by simple string matching over ingredient lists. The knowledge graph is a Phase 2 addition that will significantly expand the set of retrievable recipes without requiring the user to perfectly describe their inventory.

### Context graph (household behaviour trace)
A temporal graph of cooking behaviour, where nodes are sessions and edges encode relationships:

```
Session(2026-05-03) ──followed_by──▶ Session(2026-05-04)
Session(2026-05-03) ──used_protein──▶ chicken
Session(2026-05-03) ──feedback──▶ Recipe("Chicken Stir-fry", thumbs_up)
Recipe("Chicken Stir-fry") ──contains──▶ sriracha, hoisin, sesame_oil
```

**MVP status:** partially implemented. The `data/episodic_memory.json` file is effectively a flattened context graph — it captures the temporal sequence of sessions, proteins used, recipes shown, and feedback. What it lacks is the graph traversal interface: in the MVP, the system does a linear scan of the last 7 entries to enforce nutritional diversity. In the full design, this becomes a graph query: *"find all proteins used in the last 7 sessions"* or *"find sessions where similar-cuisine dishes were liked"*.

---

## 7. Cost Analysis

### Per session (MVP)
| Item | Estimate |
|---|---|
| System prompt (profile + history injection) | ~1,500 tokens input |
| ReAct loop (tool calls + results) | ~2,000 tokens input |
| Final recommendations output | ~400 tokens output |
| Reflection node (every 5 sessions) | ~3,000 tokens input + 300 output |
| **Total per session** | **~3,900 tokens input / 400 tokens output** |
| **Cost at Claude Sonnet 4.6 pricing ($3/MTok in, $15/MTok out)** | **~$0.018/session** |

### Monthly (30 sessions / month, 1 household)
| Item | Cost |
|---|---|
| Claude API | ~$0.54 |
| LangSmith tracing | Free (5,000 traces/month on Developer plan) |
| LangGraph Platform (local dev) | Free |
| LangGraph Cloud (if deployed) | Free up to 100k node executions |
| **Total** | **~$0.54/month** |

### Phase 2 additions
| Addition | Cost impact |
|---|---|
| Embedding 10k recipes (one-time) | ~$0.10 at text-embedding-3-small |
| Vector DB (Pinecone free tier) | $0 up to 1M vectors |
| Cross-encoder re-ranking | +~500 tokens per query (~$0.005/session) |
| **Revised monthly cost** | **~$0.70/month** |

The system is designed to be extremely cost-efficient by design: prompt caching for semantic memory (pays on first call, $0 thereafter), lazy instruction fetching (only retrieve what is selected), and a single-agent architecture that minimises redundant LLM calls.

---

## 8. Evaluation and the Data Flywheel

The system includes a lightweight evaluation loop built into the product:

**Primary metric:** `thumbs_up_rate` — the fraction of shown recipes that received a thumbs up, tracked per session, per cuisine, and per protein type. Written to `data/metrics.json` after every session.

**How the signal closes the loop:**

| Signal | Interpretation | Automated response |
|---|---|---|
| Overall rate < 50% | Recommendations are generally off | Check ReAct reasoning in LangSmith trace |
| Specific protein always thumbs-down | Unlisted dislike detected | Auto-append to dislikes after 2+ consecutive downs |
| Specific cuisine always thumbs-down | Style mismatch | Add to procedural rules via reflection |
| All 3 recommendations thumbs-down in one session | Complete miss | In-session retry with a different search strategy (up to 2×) |
| Rate improving across sessions | Learning is working | Signal to expand corpus / add more nuance |

The reflection node is the automated version of this loop — it runs every 5 sessions, reads the entire episodic history, and rewrites the procedural memory. No manual prompt engineering is required as the system scales.

---

## 9. Roadmap

| Phase | What gets added | Trigger |
|---|---|---|
| **Phase 1 (built)** | Single-agent MVP, CoALA memory, TheMealDB, feedback loop, reflection | — |
| **Phase 2** | Full RAG pipeline (Recipe1M+, hybrid retrieval, cross-encoder re-ranker), knowledge graph (USDA FoodData Central) | Once ~20 sessions of feedback exist to guide corpus curation |
| **Phase 3** | Decompose into multi-agent supervisor (Orchestrator, Inventory, Retrieval, Evaluator, Personalization) with ReAct + Tree of Thoughts | When retrieval and evaluation become independently optimisable; when multi-tenant is needed |
| **Phase 4** | Proactive notifications (*"you have salmon expiring tomorrow"*), weekly meal planning, grocery ordering integration | When the household profile is stable and recommendation quality is consistently high |

---

## 10. Summary

Tonight's Dinner is a household dinner recommendation agent built on LangGraph with a principled implementation of CoALA memory, a clear RAG design, and an ingredient knowledge graph — currently at the MVP stage where the feedback loop is running and the data flywheel is beginning to turn.

The single-agent architecture was chosen deliberately: the task is sequential, the data is sparse, and the bottleneck at this stage is preference signal, not reasoning capacity. The multi-agent supervisor design is the explicit next step, with clear architectural interfaces already drawn and a defined trigger for when to make the transition.

The system costs approximately $0.54/month to run for a single household — significantly less than a single takeout order — and gets cheaper per session as prompt caching takes effect and the procedural memory stabilises.
