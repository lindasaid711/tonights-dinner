import json
import re
import datetime
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import interrupt

from src.state import AgentState
from src.memory import load_json, save_json, PROFILE_FILE, EPISODIC_FILE, METRICS_FILE
from src.tools import search_recipes, filter_by_constraints, parse_receipt_image, _recipe_cache

TWENTY_FOODS = [
    "Chicken", "Beef", "Salmon", "Pasta", "Rice",
    "Pizza", "Tacos", "Curry", "Stir-fry", "Soup",
    "Salad", "Steak", "Shrimp", "Tofu", "Eggs",
    "Lentils", "Pork", "Lamb", "Burgers", "Sushi",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_system_prompt(profile: dict, inventory: str, recent_history: list) -> str:
    allergies = profile.get("allergies", "none")
    dislikes = profile.get("dislikes", "none")
    top_3 = ", ".join(profile.get("top_3_foods", [])) or "not specified"
    rules_list = profile.get("procedural_rules", [])
    rules = "\n".join(f"- {r}" for r in rules_list) if rules_list else "None yet."

    history_lines = []
    for entry in recent_history:
        date = entry.get("date", "")
        proteins = [r.get("primary_protein", "unknown") for r in entry.get("recommendations", [])]
        history_lines.append(f"  {date}: {', '.join(proteins)}")
    history_str = "\n".join(history_lines) if history_lines else "  No history yet — this is the first session."

    return f"""You are a dinner recommendation agent for a single household. Your job is to find 3 recipes for tonight that the household will genuinely want to cook and eat.

## Household Profile
- Allergies (HARD constraint — never violate): {allergies}
- Dislikes: {dislikes}
- Favourite foods: {top_3}

## Tonight's Available Ingredients
{inventory}

## Learned Rules (derived from past feedback — follow strictly)
{rules}

## Recent Dinner History — last 7 days
{history_str}

## CRITICAL: Nutritional Diversity
- Do NOT recommend any recipe whose primary protein matches any protein in the recent dinner history.
- All 3 recommendations must use different primary proteins from each other.

## Your reasoning process — follow these steps in order

**Step 1: Reason about the inventory using your own culinary knowledge.**
Look at "Tonight's Available Ingredients" and think: what 5–6 meals could realistically and deliciously be made from these ingredients? Use your knowledge as a chef to generate candidates. Consider:
- What is the main protein or base ingredient?
- What cuisines naturally use these combinations?
- What would actually taste good together tonight?
Do NOT call any tools yet. Just think.

**Step 2: Look up each candidate in TheMealDB by name.**
For each meal you thought of, call search_recipes() using that specific meal name as the query — not the ingredients.
For example: search_recipes("lemon chicken"), search_recipes("pasta arrabiata"), search_recipes("beef stir fry").
- If TheMealDB returns the meal, use its ingredient list as the reference for the recipe.
- If TheMealDB does not return it, you may still recommend it using your own knowledge — note "recipe from AI knowledge" in the substitutions field.

**Step 3: Filter and check constraints.**
Call filter_by_constraints on your candidates to remove any that contain allergens or dislikes.
Verify the final 3 picks have different primary proteins and none match the recent dinner history.

**Step 4: Check substitutions.**
For each final recipe, compare its ingredient list against tonight's inventory.
- If a minor ingredient is missing, suggest a logical culinary swap (e.g. white wine → lemon juice + stock, fresh herbs → dried herbs, butter → olive oil).
- If a core ingredient is missing but could be substituted (e.g. chicken thighs instead of chicken breast), note it.
- Never recommend a recipe whose main protein cannot be found or substituted from the inventory.

**Step 5: Always return exactly 3 recipes.** Never give up or return fewer.

When done, output ONLY a valid JSON array with no surrounding text:
[
  {{"name": "Recipe Name", "primary_protein": "chicken", "key_ingredients": ["chicken", "garlic", "lemon"], "cook_time_min": 30, "substitutions": []}},
  {{"name": "Recipe Name 2", "primary_protein": "beef", "key_ingredients": ["beef", "onion", "tomato"], "cook_time_min": 45, "substitutions": ["white wine → lemon juice"]}},
  {{"name": "Recipe Name 3", "primary_protein": "vegetarian", "key_ingredients": ["pasta", "basil", "parmesan"], "cook_time_min": 20, "substitutions": ["fresh basil → dried basil"]}}
]"""


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b["text"] if isinstance(b, dict) and b.get("type") == "text"
            else (b.text if hasattr(b, "text") else str(b))
            for b in content
        )
    return str(content)


def _parse_recommendations(content: str) -> list:
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _parse_feedback(raw: str, recommendations: list) -> dict:
    tokens = re.split(r'[\s,;]+', raw.lower().strip())
    signals = []
    for token in tokens:
        if token in ('👍', 'up', 'yes', 'y', 'good', 'like', '1', 'thumbsup'):
            signals.append('up')
        elif token in ('👎', 'down', 'no', 'n', 'bad', 'dislike', '0', 'thumbsdown'):
            signals.append('down')

    feedback = {}
    for i, rec in enumerate(recommendations):
        feedback[rec['name']] = signals[i] if i < len(signals) else 'up'
    return feedback


# ── Onboarding nodes ──────────────────────────────────────────────────────────

def load_context(state: AgentState) -> dict:
    profile = load_json(PROFILE_FILE, {})
    episodic = load_json(EPISODIC_FILE, [])
    recent = sorted(episodic, key=lambda x: x.get("date", ""), reverse=True)[:7]
    return {
        "user_profile": profile,
        "recent_history": recent,
        "onboarding_complete": "allergies" in profile,
        "retry_count": 0,
    }


def ask_allergies(state: AgentState) -> dict:
    response = interrupt("Do you have any food allergies? (e.g. 'nuts, shellfish, dairy' — or type 'none')")
    return {"user_profile": {**state.get("user_profile", {}), "allergies": response}}


def ask_dislikes(state: AgentState) -> dict:
    response = interrupt("Any foods you strongly dislike? (e.g. 'mushrooms, olives' — or type 'none')")
    return {"user_profile": {**state.get("user_profile", {}), "dislikes": response}}


def ask_preferences(state: AgentState) -> dict:
    food_list = "\n".join(f"  {i+1:2}. {food}" for i, food in enumerate(TWENTY_FOODS))
    response = interrupt(
        f"From the list below, enter the numbers of your top 3 favourite dinner types "
        f"(e.g. '1, 5, 12'):\n\n{food_list}"
    )
    selected = []
    for part in re.split(r'[\s,;]+', response):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(TWENTY_FOODS) and len(selected) < 3:
                selected.append(TWENTY_FOODS[idx])
        except ValueError:
            pass
    return {"user_profile": {**state.get("user_profile", {}), "top_3_foods": selected}}


def ask_inventory(state: AgentState) -> dict:
    response = interrupt(
        "What ingredients do you have available tonight?\n"
        "Describe what's in your fridge/pantry (e.g. 'chicken breast, garlic, pasta, tomatoes').\n"
        "Or paste a grocery receipt photo as base64, prefixed with 'PHOTO:'"
    )
    if response.strip().upper().startswith("PHOTO:"):
        image_b64 = response[6:].strip()
        # Detect media type from base64 header bytes
        import binascii
        try:
            header = binascii.a2b_base64(image_b64[:16])
            if header[:4] == b'RIFF':
                media_type = "image/webp"
            elif header[:8] == b'\x89PNG\r\n\x1a\n':
                media_type = "image/png"
            else:
                media_type = "image/jpeg"
        except Exception:
            media_type = "image/webp"
        inventory = parse_receipt_image(image_b64, media_type=media_type)
    else:
        inventory = response
    return {"inventory": inventory}


def save_profile(state: AgentState) -> dict:
    save_json(PROFILE_FILE, state.get("user_profile", {}))
    return {"onboarding_complete": True}


# ── ReAct + recommendation nodes ──────────────────────────────────────────────

def react_agent(state: AgentState) -> dict:
    profile = load_json(PROFILE_FILE, state.get("user_profile", {}))
    system_prompt = _build_system_prompt(
        profile,
        state.get("inventory", "not specified"),
        state.get("recent_history", []),
    )

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=4096)
    llm_with_tools = llm.bind_tools([search_recipes, filter_by_constraints])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please find me 3 dinner recipes for tonight."),
    ]
    response = None

    for _ in range(12):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            if tc["name"] == "search_recipes":
                result = search_recipes.invoke(tc["args"])
            elif tc["name"] == "filter_by_constraints":
                result = filter_by_constraints.invoke(tc["args"])
            else:
                result = f"unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    final_content = _extract_text(response.content) if response else ""
    recommendations = _parse_recommendations(final_content)

    return {
        "recommendations": recommendations,
        "messages": [AIMessage(content=final_content)] if final_content else [],
    }


def present_recommendations(state: AgentState) -> dict:
    recs = state.get("recommendations", [])
    if not recs:
        return {"messages": [AIMessage(
            content="I wasn't able to find 3 suitable recipes. Let's try again with different constraints."
        )]}

    lines = ["Here are tonight's dinner recommendations:\n"]
    for i, rec in enumerate(recs, 1):
        ingredients = ", ".join(rec.get("key_ingredients", [])[:5])
        lines.append(f"**{i}. {rec['name']}**")
        lines.append(f"   Protein: {rec.get('primary_protein', 'N/A')}  |  Cook time: ~{rec.get('cook_time_min', '?')} min")
        lines.append(f"   Ingredients: {ingredients}\n")

    return {"messages": [AIMessage(content="\n".join(lines))]}


def collect_feedback(state: AgentState) -> dict:
    recs = state.get("recommendations", [])
    names = "\n".join(f"  {i+1}. {r['name']}" for i, r in enumerate(recs))

    raw = interrupt(f"Rate each recipe 👍 or 👎 (e.g. '👍 👎 👍' or 'up down up'):\n{names}")
    feedback = _parse_feedback(raw, recs)

    retry_count = state.get("retry_count", 0)
    if feedback and all(v == "down" for v in feedback.values()):
        retry_count += 1

    return {"feedback": feedback, "retry_count": retry_count}


def save_feedback(state: AgentState) -> dict:
    today = datetime.date.today().isoformat()

    episodic = load_json(EPISODIC_FILE, [])
    entry = {
        "date": today,
        "recommendations": state.get("recommendations", []),
        "feedback": state.get("feedback", {}),
    }
    reasons = state.get("feedback_reasons", {})
    if any(v for v in reasons.values()):
        entry["feedback_reasons"] = {k: v for k, v in reasons.items() if v}
    episodic.append(entry)
    save_json(EPISODIC_FILE, episodic)

    metrics = load_json(METRICS_FILE, {
        "total_recommendations": 0,
        "total_thumbs_up": 0,
        "thumbs_up_rate": 0.0,
        "sessions": 0,
    })
    feedback = state.get("feedback", {})
    n_recs = len(state.get("recommendations", []))
    n_up = sum(1 for v in feedback.values() if v == "up")

    metrics["total_recommendations"] += n_recs
    metrics["total_thumbs_up"] += n_up
    metrics["sessions"] += 1
    if metrics["total_recommendations"] > 0:
        metrics["thumbs_up_rate"] = round(
            metrics["total_thumbs_up"] / metrics["total_recommendations"], 3
        )
    save_json(METRICS_FILE, metrics)

    return {}


def reflect(state: AgentState) -> dict:
    """Runs every 5 sessions: reads feedback history and updates procedural rules."""
    metrics = load_json(METRICS_FILE, {})
    sessions = metrics.get("sessions", 0)

    if sessions == 0 or sessions % 5 != 0:
        return {}

    episodic = load_json(EPISODIC_FILE, [])
    if not episodic:
        return {}

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)
    prompt = (
        "Review this cooking feedback history and write 3-5 concise, actionable rules "
        "for improving future recipe recommendations.\n\n"
        f"History:\n{json.dumps(episodic, indent=2)}\n\n"
        "Rules should be short statements. Examples:\n"
        '- "Avoid fish dishes — consistently low feedback"\n'
        '- "Prefer pasta dishes under 30 minutes"\n'
        '- "User enjoys Asian cuisine"\n\n'
        "Return ONLY a JSON array of rule strings, no other text."
    )

    response = llm.invoke([
        SystemMessage(content="You are a pattern recognition assistant for a cooking agent."),
        HumanMessage(content=prompt),
    ])

    content = _extract_text(response.content)
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        try:
            rules = json.loads(match.group())
            profile = load_json(PROFILE_FILE, {})
            profile["procedural_rules"] = rules
            save_json(PROFILE_FILE, profile)
        except json.JSONDecodeError:
            pass

    return {}
