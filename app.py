from dotenv import load_dotenv
load_dotenv()

import re
import json
import time
import base64

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.nodes import (
    load_context,
    save_feedback,
    reflect,
    _build_system_prompt,
    _parse_recommendations,
    _extract_text,
    TWENTY_FOODS,
)
from src.tools import search_recipes, filter_by_constraints, parse_receipt_image
from src.memory import load_json, save_json, PROFILE_FILE, EPISODIC_FILE, METRICS_FILE


# ── Page config ───────────────────────────────────────────────────────────────
# This is the very first Streamlit call — sets the browser tab title and layout.
# "wide" gives us room for the sidebar + main chat panel side by side.

st.set_page_config(
    page_title="Tonight's Dinner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ─────────────────────────────────────────────────────────────
# Streamlit re-runs the entire script from top to bottom every time the user
# does anything (types a message, clicks a button). Session state is a
# dictionary that survives those re-runs — it's how we remember where we are
# in the conversation.
#
# flow_stage  — tracks which step of the conversation we're on
# messages    — the full chat history displayed on screen
# agent_state — data the agent needs (profile, inventory, recommendations, etc.)
# tool_call_log — what the agent searched for, shown in the expander

if "flow_stage" not in st.session_state:
    st.session_state.flow_stage = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {}
if "tool_call_log" not in st.session_state:
    st.session_state.tool_call_log = []
if "feedback_votes" not in st.session_state:
    st.session_state.feedback_votes = {}
if "feedback_reasons" not in st.session_state:
    st.session_state.feedback_reasons = {}
if "profile_cache" not in st.session_state:
    st.session_state.profile_cache = load_json(PROFILE_FILE, {})
if "selected_foods" not in st.session_state:
    st.session_state.selected_foods = []

FOOD_EMOJIS = {
    "Chicken": "🍗", "Beef": "🥩", "Salmon": "🐟", "Pasta": "🍝",
    "Rice": "🍚", "Pizza": "🍕", "Tacos": "🌮", "Curry": "🍛",
    "Stir-fry": "🥘", "Soup": "🍜", "Salad": "🥗", "Steak": "🥩",
    "Shrimp": "🍤", "Tofu": "🫘", "Eggs": "🍳", "Lentils": "🫘",
    "Pork": "🍖", "Lamb": "🍖", "Burgers": "🍔", "Sushi": "🍱",
}


# ── Helper: add a message to chat history ─────────────────────────────────────
# Two small helpers so we don't repeat this pattern everywhere.

def add_assistant_msg(text: str):
    st.session_state.messages.append({"role": "assistant", "content": text})

def add_user_msg(text: str):
    st.session_state.messages.append({"role": "user", "content": text})


# ── ReAct agent with tool call capture ───────────────────────────────────────
# This is the brain of the app. It runs the same ReAct reasoning loop as the
# original agent but also records every tool call so we can display them in
# the "See how the agent found these" expander.
#
# ReAct = Reason → Act → Observe → repeat.
# The agent thinks about what to search, calls the search tool, reads the
# results, thinks again, and keeps going until it has 3 valid recipes.

def run_react_agent(state: dict, status=None) -> dict:
    profile = load_json(PROFILE_FILE, state.get("user_profile", {}))
    system_prompt = _build_system_prompt(
        profile,
        state.get("inventory", "not specified"),
        state.get("recent_history", []),
    )

    def invoke_with_retry(llm_with_tools, messages, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                return llm_with_tools.invoke(messages)
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** attempt)

    # Haiku is 5x faster than Sonnet for tool-calling loops — sufficient for
    # recipe search and constraint filtering. Sonnet is reserved for reflect.
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048)
    llm_with_tools = llm.bind_tools([search_recipes, filter_by_constraints])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please find me 3 dinner recipes for tonight."),
    ]

    tool_calls_log = []
    response = None

    for _ in range(7):
        response = invoke_with_retry(llm_with_tools, messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            try:
                if tc["name"] == "search_recipes":
                    args = tc["args"]
                    query = args.get("query", "") if isinstance(args, dict) else str(args)
                    result = search_recipes.invoke({"query": str(query)})
                elif tc["name"] == "filter_by_constraints":
                    args = tc["args"]
                    if isinstance(args, dict):
                        names = args.get("recipe_names", [])
                        excluded = args.get("excluded_ingredients", [])
                        if isinstance(names, str):
                            names = [names]
                        if isinstance(excluded, str):
                            excluded = [excluded]
                    else:
                        names, excluded = [], []
                    result = filter_by_constraints.invoke({"recipe_names": names, "excluded_ingredients": excluded})
                else:
                    result = f"unknown tool: {tc['name']}"
            except Exception as e:
                result = f"tool error: {str(e)}"

            tool_calls_log.append({
                "tool": tc["name"],
                "args": tc["args"],
                "result_preview": str(result)[:400],
            })

            if status:
                if tc["name"] == "search_recipes":
                    status.write(f"🔍 Searched for **\"{tc['args'].get('query', '')}\"**")
                elif tc["name"] == "filter_by_constraints":
                    excluded = tc["args"].get("excluded_ingredients", [])
                    kept = len(result) if isinstance(result, list) else "?"
                    status.write(f"🔎 Filtered out allergens/dislikes — {kept} recipes remaining")

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    st.session_state.tool_call_log = tool_calls_log
    final_content = _extract_text(response.content) if response else ""
    recommendations = _parse_recommendations(final_content)

    return {
        "recommendations": recommendations,
        "messages": [AIMessage(content=final_content)] if final_content else [],
    }


# ── Sidebar: live memory state ────────────────────────────────────────────────
# The sidebar is the "show your work" panel for the demo. It reads directly
# from the JSON files on disk every time the page refreshes — so when the
# reflect node writes new procedural rules after session 5, they appear here
# automatically without any extra code.

with st.sidebar:
    st.title("Memory State")
    st.caption("Updates live after each session")

    profile = st.session_state.profile_cache

    st.subheader("Household Profile")
    st.markdown(f"**Allergies:** {profile.get('allergies', 'not set')}")
    st.markdown(f"**Dislikes:** {profile.get('dislikes', 'not set')}")
    top3 = ", ".join(profile.get("top_3_foods", [])) or "not set"
    st.markdown(f"**Favourite foods:** {top3}")

    with st.expander("✏️ Edit profile"):
        new_allergies = st.text_input(
            "Allergies",
            value=profile.get("allergies", ""),
            placeholder="e.g. gluten, nuts, dairy — or none",
            key="edit_allergies",
        )
        new_dislikes = st.text_input(
            "Dislikes",
            value=profile.get("dislikes", ""),
            placeholder="e.g. mushrooms, olives — or none",
            key="edit_dislikes",
        )
        new_top3 = st.text_input(
            "Favourite foods (comma-separated)",
            value=", ".join(profile.get("top_3_foods", [])),
            placeholder="e.g. Chicken, Pasta, Curry",
            key="edit_top3",
        )
        if st.button("Save changes", key="save_profile_btn", type="primary"):
            profile["allergies"] = new_allergies.strip()
            profile["dislikes"] = new_dislikes.strip()
            profile["top_3_foods"] = [f.strip() for f in new_top3.split(",") if f.strip()]
            save_json(PROFILE_FILE, profile)
            # Update session state immediately so sidebar reflects changes without delay
            st.session_state.profile_cache = profile
            st.success("✅ Profile updated!")

    st.divider()
    st.subheader("Learned Rules")
    rules = profile.get("procedural_rules", [])
    if rules:
        for rule in rules:
            st.markdown(f"- {rule}")
    else:
        st.caption("None yet — appear automatically after session 5")

    st.divider()
    metrics = load_json(METRICS_FILE, {})
    prev_sessions = st.session_state.get("prev_sessions", metrics.get("sessions", 0))
    sessions_now = metrics.get("sessions", 0)
    session_delta = sessions_now - prev_sessions if sessions_now > prev_sessions else None
    st.session_state.prev_sessions = sessions_now

    col1, col2 = st.columns(2)
    col1.metric("Sessions", sessions_now, delta=f"+{session_delta}" if session_delta else None)
    rate = metrics.get("thumbs_up_rate", 0.0)
    col2.metric("Thumbs up", f"{rate:.0%}")

    st.divider()
    with st.expander("🎮 Demo controls"):
        st.caption("Use these to reset the app state during a demo.")
        if st.button("▶ Show onboarding from scratch", key="reset_onboarding"):
            save_json(PROFILE_FILE, {})
            st.session_state.profile_cache = {}
            st.session_state.flow_stage = None
            st.session_state.messages = []
            st.session_state.agent_state = {}
            st.session_state.tool_call_log = []
            st.rerun()
        if st.button("↺ Reset to 4 seed sessions", key="reset_sessions"):
            # Re-run seed logic inline without subprocess
            from seed_demo import SEED_SESSIONS, compute_metrics
            save_json(EPISODIC_FILE, SEED_SESSIONS)
            save_json(METRICS_FILE, compute_metrics(SEED_SESSIONS))
            st.success("Reset to 4 sessions — ready for live demo session 5.")
            st.rerun()


# ── Main panel title ──────────────────────────────────────────────────────────

st.title("🍽️ Tonight's Dinner")
st.caption("Your personal dinner agent — learns your household's taste over time.")

# Show onboarding progress bar
onboarding_steps = {
    "onboarding_allergies": (1, "Step 1 of 4 — Allergies"),
    "onboarding_dislikes": (2, "Step 2 of 4 — Dislikes"),
    "onboarding_preferences": (3, "Step 3 of 4 — Favourite foods"),
    "onboarding_inventory": (4, "Step 4 of 4 — Tonight's ingredients"),
    "confirming_inventory": (4, "Step 4 of 4 — Confirm ingredients"),
}
if st.session_state.flow_stage in onboarding_steps:
    step_num, label = onboarding_steps[st.session_state.flow_stage]
    st.progress(step_num / 4, text=label)


# ── Bootstrap ────────────────────────────────────────────────────────────────
# This block runs exactly once — when flow_stage is None (a brand new browser
# session). It calls load_context to check if the user has already completed
# onboarding (i.e. does user_profile.json exist on disk?).
#
# If yes → skip straight to asking for tonight's inventory.
# If no  → start onboarding from allergies.
#
# st.rerun() tells Streamlit to immediately re-run the script from the top.
# We use it here so the first assistant message appears right away.

if st.session_state.flow_stage is None:
    result = load_context({})
    st.session_state.agent_state.update(result)
    if result.get("onboarding_complete"):
        st.session_state.flow_stage = "onboarding_inventory"
        add_assistant_msg(
            "Welcome back! What ingredients do you have available tonight?\n\n"
            "List what's in your fridge and pantry (e.g. *chicken breast, garlic, pasta, tomatoes*), "
            "or upload a photo of your grocery receipt below."
        )
    else:
        st.session_state.flow_stage = "onboarding_allergies"
        add_assistant_msg(
            "Welcome to Tonight's Dinner! Let's set up your profile — this only takes a minute.\n\n"
            "**Do you have any food allergies?** (e.g. *nuts, shellfish, gluten* — or type *none*)"
        )
    st.rerun()


# ── Auto-run the agent ────────────────────────────────────────────────────────
# When flow_stage is "recommending", the agent runs automatically — no user
# input needed. This block fires at the top of the script before anything is
# drawn, so the spinner appears while the agent is thinking.
#
# After it finishes, we store the results, add the recommendation message to
# chat history, then flip the stage to "collecting_feedback" and rerun so
# the chat history and the feedback prompt both render.

if st.session_state.flow_stage == "recommending":
    with st.status("🍽️ Finding tonight's recipes...", expanded=True) as status:
        status.write("Reading your inventory and household profile...")
        result = run_react_agent(st.session_state.agent_state, status)
        st.session_state.agent_state.update(result)

        recs = result.get("recommendations", [])

        # Fallback: if inventory-based search returned nothing, retry without inventory constraint
        if not recs:
            status.write("⚠️ Couldn't find recipes from your inventory — retrying with broader search...")
            fallback_state = {**st.session_state.agent_state, "inventory": "anything — suggest popular, easy weeknight dinners"}
            result = run_react_agent(fallback_state, status)
            recs = result.get("recommendations", [])
            st.session_state.agent_state["recommendations"] = recs

        if recs:
            status.update(label="✅ Found 3 recipes!", state="complete", expanded=False)
            lines = ["Here are tonight's dinner recommendations:\n"]
            for i, rec in enumerate(recs, 1):
                ingredients = ", ".join(rec.get("key_ingredients", [])[:5])
                lines.append(f"**{i}. {rec['name']}**")
                lines.append(
                    f"Protein: {rec.get('primary_protein', 'N/A')}  |  "
                    f"Cook time: ~{rec.get('cook_time_min', '?')} min"
                )
                lines.append(f"Ingredients: {ingredients}")
                subs = rec.get("substitutions", [])
                if subs:
                    lines.append(f"*Swaps: {', '.join(subs)}*")
                lines.append("")
            add_assistant_msg("\n".join(lines))
        else:
            status.update(label="⚠️ Couldn't find recipes", state="error", expanded=False)
            add_assistant_msg(
                "I wasn't able to find suitable recipes right now — TheMealDB may be having issues. "
                "Try typing your ingredients manually in the next session."
            )

        add_assistant_msg("How do these sound? Rate each one below:")
        st.session_state.feedback_votes = {}
        st.session_state.feedback_reasons = {}
        st.session_state.flow_stage = "collecting_feedback"
    st.rerun()


# ── Render chat history ───────────────────────────────────────────────────────
# Loop through every message stored in session state and draw it as a chat
# bubble. st.chat_message("assistant") gives a bubble on the left;
# st.chat_message("user") gives one on the right.

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Tool call expander ────────────────────────────────────────────────────────
# Shows after recommendations appear, while we're waiting for feedback.
# This is the demo moment — the class can see the agent's reasoning: what
# queries it tried, what came back, how it filtered.

if st.session_state.tool_call_log and st.session_state.flow_stage == "collecting_feedback":
    with st.expander("🔍 See how the agent found these recipes"):
        for i, call in enumerate(st.session_state.tool_call_log, 1):
            st.markdown(f"**Step {i}: `{call['tool']}`**")
            st.json(call["args"])
            if call["tool"] == "search_recipes":
                try:
                    preview = json.loads(call["result_preview"])
                    names_found = [r["name"] for r in preview[:5]]
                    st.caption(f"Found: {', '.join(names_found)}")
                except Exception:
                    st.caption(call["result_preview"])
            else:
                st.caption(f"Kept: {call['result_preview']}")
            st.divider()


# ── Feedback buttons ─────────────────────────────────────────────────────────
# When we're collecting feedback, show a thumbs up / thumbs down button for
# each recipe instead of asking the user to type. Each click saves that vote
# to session state immediately. Once all recipes have a vote, a Submit button
# appears and we process + save the feedback.

if st.session_state.flow_stage == "collecting_feedback":
    recs = st.session_state.agent_state.get("recommendations", [])
    votes = st.session_state.feedback_votes

    st.markdown("---")
    for rec in recs:
        name = rec["name"]
        current = votes.get(name)
        col_name, col_up, col_down = st.columns([5, 1, 1])
        col_name.markdown(f"**{name}**")

        up_style = "primary" if current == "up" else "secondary"
        down_style = "primary" if current == "down" else "secondary"

        if col_up.button("👍", key=f"up_{name}", type=up_style):
            st.session_state.feedback_votes[name] = "up"
            st.session_state.feedback_reasons.pop(name, None)
            st.rerun()
        if col_down.button("👎", key=f"down_{name}", type=down_style):
            st.session_state.feedback_votes[name] = "down"
            st.rerun()

        # Show reason text box only when thumbs down is selected.
        # No value= override — Streamlit holds the typed text in session_state
        # automatically via the key, so it persists across reruns until Submit.
        if current == "down":
            st.text_input(
                "What didn't work? (optional — included when you submit below)",
                key=f"reason_{name}",
                placeholder="e.g. don't have the ingredients, too long, not in the mood...",
            )

    all_voted = len(votes) == len(recs) and len(recs) > 0
    if all_voted:
        if st.button("Submit feedback →", type="primary"):
            feedback = votes
            # Collect any typed reasons directly from Streamlit's widget state
            reasons = {
                rec["name"]: st.session_state.get(f"reason_{rec['name']}", "")
                for rec in recs
                if st.session_state.get(f"reason_{rec['name']}", "")
            }
            st.session_state.feedback_reasons = reasons
            retry_count = st.session_state.agent_state.get("retry_count", 0)
            all_down = all(v == "down" for v in feedback.values())

            if all_down and retry_count < 2:
                st.session_state.agent_state["retry_count"] = retry_count + 1
                st.session_state.agent_state["feedback"] = feedback
                add_assistant_msg(
                    "None of those worked — let me try again with different options."
                )
                st.session_state.feedback_votes = {}
                st.session_state.flow_stage = "recommending"
            else:
                st.session_state.agent_state["feedback"] = feedback
                st.session_state.agent_state["feedback_reasons"] = st.session_state.feedback_reasons
                save_feedback(st.session_state.agent_state)
                reflect(st.session_state.agent_state)
                # Sync profile cache so sidebar shows new learned rules immediately
                updated_profile = load_json(PROFILE_FILE, {})
                st.session_state.profile_cache = updated_profile
                if updated_profile.get("procedural_rules"):
                    st.balloons()
                add_assistant_msg(
                    "Thanks for the feedback — saved! Check the sidebar to see your updated "
                    "session count and any new learned rules."
                )
                st.session_state.feedback_votes = {}
                st.session_state.flow_stage = "done"
            st.rerun()
    st.markdown("---")


# ── Receipt photo uploader ────────────────────────────────────────────────────
# Only shown on the inventory step. The user can either type their ingredients
# or upload a photo of their grocery receipt. Claude's vision capability reads
# the photo and extracts the ingredient list automatically.

if st.session_state.flow_stage == "onboarding_inventory":
    uploaded = st.file_uploader(
        "Or upload a grocery receipt photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        raw_bytes = uploaded.read()
        image_b64 = base64.b64encode(raw_bytes).decode()
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        media_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
        }.get(ext, "image/jpeg")
        with st.spinner("Reading your receipt..."):
            inventory = parse_receipt_image(image_b64, media_type=media_type)
        add_user_msg("[Receipt photo uploaded]")
        add_assistant_msg(
            "Here's what I found in your receipt — remove anything that isn't a food ingredient "
            "and add anything that's missing, then hit **Confirm** to find recipes:"
        )
        st.session_state.agent_state["inventory"] = inventory
        st.session_state.flow_stage = "confirming_inventory"
        st.rerun()


# ── Food preference card picker ───────────────────────────────────────────────
# Shown during onboarding step 3. Displays all 20 foods as clickable emoji cards
# in a 4-column grid. Selected cards highlight in orange. Confirm button appears
# once exactly 3 are chosen.

if st.session_state.flow_stage == "onboarding_preferences":
    selected = st.session_state.selected_foods
    st.markdown(f"**{len(selected)}/3 selected**")

    cols = st.columns(4)
    for i, food in enumerate(TWENTY_FOODS):
        emoji = FOOD_EMOJIS.get(food, "🍽️")
        label = f"{emoji} {food}"
        is_selected = food in selected
        btn_type = "primary" if is_selected else "secondary"
        if cols[i % 4].button(label, key=f"food_{food}", type=btn_type, use_container_width=True):
            if is_selected:
                st.session_state.selected_foods.remove(food)
            elif len(selected) < 3:
                st.session_state.selected_foods.append(food)
            st.rerun()

    if len(selected) == 3:
        st.markdown(f"**Selected:** {' · '.join(selected)}")
        if st.button("Confirm →", type="primary"):
            profile = st.session_state.agent_state.get("user_profile", {})
            profile["top_3_foods"] = selected
            st.session_state.agent_state["user_profile"] = profile
            save_json(PROFILE_FILE, profile)
            st.session_state.profile_cache = profile
            st.session_state.agent_state["onboarding_complete"] = True
            st.session_state.selected_foods = []
            st.session_state.flow_stage = "onboarding_inventory"
            add_assistant_msg(
                "Perfect — profile saved! **What ingredients do you have available tonight?**\n\n"
                "List what's in your fridge and pantry, or upload a receipt photo below."
            )
            st.rerun()


# ── Inventory confirmation ────────────────────────────────────────────────────
# Shows the parsed/typed inventory in an editable text area. The user can
# clean it up (remove non-food items from a receipt, fix typos, add things)
# before the agent runs. Confirm button triggers the search.

if st.session_state.flow_stage == "confirming_inventory":
    current_inventory = st.session_state.agent_state.get("inventory", "")
    edited = st.text_area(
        "Your ingredients:",
        value=current_inventory,
        height=150,
        key="inventory_edit",
        label_visibility="collapsed",
    )
    if st.button("Confirm — find my recipes →", type="primary"):
        st.session_state.agent_state["inventory"] = edited
        add_user_msg(f"Confirmed inventory: {edited}")
        st.session_state.flow_stage = "recommending"
        st.rerun()


# ── Done state ────────────────────────────────────────────────────────────────
# After feedback is saved, show a success message and a button to start fresh.
# st.stop() prevents the chat input from appearing — there's nothing left to
# type at this point.

if st.session_state.flow_stage == "done":
    st.success("Session saved! Your feedback helps me get better over time.")
    if st.button("Start a new session"):
        st.session_state.flow_stage = None
        st.session_state.messages = []
        st.session_state.agent_state = {}
        st.session_state.tool_call_log = []
        st.rerun()
    st.stop()


# ── Chat input ────────────────────────────────────────────────────────────────
# st.chat_input() draws the text box at the bottom of the screen.
# It returns whatever the user typed, or None if they haven't typed anything.
# When it returns a value, we figure out what stage we're on and handle it.

user_input = st.chat_input("Type your response here...")

if user_input:
    add_user_msg(user_input)
    stage = st.session_state.flow_stage
    profile = st.session_state.agent_state.get("user_profile", {})

    # Onboarding step 1: collect allergies
    if stage == "onboarding_allergies":
        profile["allergies"] = user_input
        st.session_state.agent_state["user_profile"] = profile
        st.session_state.flow_stage = "onboarding_dislikes"
        add_assistant_msg(
            "Got it. **Any foods you strongly dislike?** "
            "(e.g. *mushrooms, olives* — or type *none*)"
        )

    # Onboarding step 2: collect dislikes, then show food card picker
    elif stage == "onboarding_dislikes":
        profile["dislikes"] = user_input
        st.session_state.agent_state["user_profile"] = profile
        st.session_state.selected_foods = []
        st.session_state.flow_stage = "onboarding_preferences"
        add_assistant_msg("Great! Pick your **top 3 favourite dinner types** from the options below:")

    # onboarding_preferences is handled by card buttons below, not chat input

    # Onboarding step 4: store inventory text, show confirmation before searching
    elif stage == "onboarding_inventory":
        st.session_state.agent_state["inventory"] = user_input
        st.session_state.flow_stage = "confirming_inventory"
        add_assistant_msg(
            "Here's what you listed — edit anything if needed, then hit **Confirm** to find recipes:"
        )

    st.rerun()
