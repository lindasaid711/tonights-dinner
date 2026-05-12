import base64
import streamlit as st
from langgraph_sdk import get_sync_client

SERVER_URL = "http://localhost:2024"
GRAPH_ID = "dinner_agent"

client = get_sync_client(url=SERVER_URL)

st.set_page_config(page_title="Tonight's Dinner", page_icon="🍽️", layout="centered")
st.title("🍽️ Tonight's Dinner")
st.caption("Tell me what's in your kitchen and I'll find tonight's recipes.")

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = client.threads.create()["thread_id"]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "started" not in st.session_state:
    st.session_state.started = False
if "done" not in st.session_state:
    st.session_state.done = False
if "awaiting_inventory" not in st.session_state:
    st.session_state.awaiting_inventory = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _thread_status(thread_id: str) -> tuple[str | None, bool]:
    """Return (interrupt_message, is_done) for the current thread state."""
    state = client.threads.get_state(thread_id)
    interrupt_msg = None
    for task in state.get("tasks", []):
        for interrupt in task.get("interrupts", []):
            val = interrupt.get("value", "")
            if isinstance(val, dict):
                parts = [val.get("prompt", "")]
                if "hint" in val:
                    parts.append(f"_{val['hint']}_")
                interrupt_msg = "\n".join(filter(None, parts))
            else:
                interrupt_msg = str(val)
            break
        if interrupt_msg:
            break
    is_done = not state.get("next") and not interrupt_msg
    return interrupt_msg, is_done


def _is_inventory_question(msg: str) -> bool:
    keywords = ("ingredient", "available tonight", "fridge", "pantry", "kitchen", "receipt")
    return any(k in msg.lower() for k in keywords)


def run_graph(resume_value=None) -> tuple[str | None, list[str], bool]:
    """Start or resume the graph. Returns (interrupt_msg, new_ai_messages, is_done)."""
    thread_id = st.session_state.thread_id
    new_ai_msgs = []

    stream_kwargs = dict(stream_mode="updates", stream_subgraphs=False)

    if resume_value is not None:
        stream = client.runs.stream(
            thread_id, GRAPH_ID,
            input=None,
            command={"resume": resume_value},
            **stream_kwargs,
        )
    else:
        stream = client.runs.stream(
            thread_id, GRAPH_ID,
            input={},
            **stream_kwargs,
        )

    for chunk in stream:
        if chunk.event != "updates":
            continue
        node_data = chunk.data.get("present_recommendations", {})
        for msg in node_data.get("messages", []):
            content = (
                msg.get("content", "") if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            if content:
                new_ai_msgs.append(content)

    interrupt_msg, is_done = _thread_status(thread_id)
    return interrupt_msg, new_ai_msgs, is_done


# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Auto-start on first load ──────────────────────────────────────────────────
if not st.session_state.started:
    st.session_state.started = True
    with st.spinner("Starting up your dinner agent…"):
        interrupt_msg, ai_msgs, done = run_graph()
    for m in ai_msgs:
        st.session_state.messages.append({"role": "assistant", "content": m})
    if interrupt_msg:
        st.session_state.messages.append({"role": "assistant", "content": interrupt_msg})
        st.session_state.awaiting_inventory = _is_inventory_question(interrupt_msg)
    st.session_state.done = done
    st.rerun()

# ── Input area ────────────────────────────────────────────────────────────────
if not st.session_state.done:

    # Show receipt uploader when the agent asks about inventory
    if st.session_state.awaiting_inventory:
        uploaded = st.file_uploader(
            "📎 Upload a grocery receipt photo (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
        )
        if uploaded is not None:
            mime = uploaded.type or "image/webp"
            b64 = base64.b64encode(uploaded.read()).decode("utf-8")
            resume_val = f"PHOTO:{b64}"
            st.session_state.messages.append({
                "role": "user",
                "content": f"📎 _{uploaded.name}_ uploaded",
            })
            st.session_state.awaiting_inventory = False
            with st.spinner("Reading your receipt…"):
                interrupt_msg, ai_msgs, done = run_graph(resume_value=resume_val)
            for m in ai_msgs:
                st.session_state.messages.append({"role": "assistant", "content": m})
            if interrupt_msg:
                st.session_state.messages.append({"role": "assistant", "content": interrupt_msg})
                st.session_state.awaiting_inventory = _is_inventory_question(interrupt_msg)
            if done:
                st.session_state.done = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "✅ All done! Your feedback has been saved. Come back tomorrow night for fresh recommendations 🍽️",
                })
            st.rerun()

    if user_input := st.chat_input("Your response… (or describe your ingredients above)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.awaiting_inventory = False
        with st.spinner("Thinking…"):
            interrupt_msg, ai_msgs, done = run_graph(resume_value=user_input)
        for m in ai_msgs:
            st.session_state.messages.append({"role": "assistant", "content": m})
        if interrupt_msg:
            st.session_state.messages.append({"role": "assistant", "content": interrupt_msg})
            st.session_state.awaiting_inventory = _is_inventory_question(interrupt_msg)
        if done:
            st.session_state.done = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "✅ All done! Your feedback has been saved. Come back tomorrow night for fresh recommendations 🍽️",
            })
        st.rerun()

else:
    st.success("Session complete!")
    if st.button("🔄 Start a new session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
