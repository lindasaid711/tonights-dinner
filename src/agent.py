from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.nodes import (
    load_context,
    ask_allergies,
    ask_dislikes,
    ask_preferences,
    ask_inventory,
    save_profile,
    react_agent,
    present_recommendations,
    collect_feedback,
    save_feedback,
    reflect,
)


def _route_after_load(state: AgentState) -> str:
    return "ask_allergies" if not state.get("onboarding_complete") else "ask_inventory"


def _route_after_feedback(state: AgentState) -> str:
    feedback = state.get("feedback", {})
    retry_count = state.get("retry_count", 0)
    all_down = bool(feedback) and all(v == "down" for v in feedback.values())
    if all_down and retry_count < 2:
        return "react_agent"
    return "save_feedback"


builder = StateGraph(AgentState)

builder.add_node("load_context", load_context)
builder.add_node("ask_allergies", ask_allergies)
builder.add_node("ask_dislikes", ask_dislikes)
builder.add_node("ask_preferences", ask_preferences)
builder.add_node("ask_inventory", ask_inventory)
builder.add_node("save_profile", save_profile)
builder.add_node("react_agent", react_agent)
builder.add_node("present_recommendations", present_recommendations)
builder.add_node("collect_feedback", collect_feedback)
builder.add_node("save_feedback", save_feedback)
builder.add_node("reflect", reflect)

builder.add_edge(START, "load_context")
builder.add_conditional_edges("load_context", _route_after_load)
builder.add_edge("ask_allergies", "ask_dislikes")
builder.add_edge("ask_dislikes", "ask_preferences")
builder.add_edge("ask_preferences", "ask_inventory")
builder.add_edge("ask_inventory", "save_profile")
builder.add_edge("save_profile", "react_agent")
builder.add_edge("react_agent", "present_recommendations")
builder.add_edge("present_recommendations", "collect_feedback")
builder.add_conditional_edges("collect_feedback", _route_after_feedback)
builder.add_edge("save_feedback", "reflect")
builder.add_edge("reflect", END)

graph = builder.compile()
