from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: dict
    inventory: str
    recommendations: list
    feedback: dict
    recent_history: list
    onboarding_complete: bool
    retry_count: int
