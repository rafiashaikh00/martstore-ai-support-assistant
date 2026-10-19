# ==========================================
# HOW THIS FILE IS CONNECTED
# ==========================================
#
#   User types a question (data_fron.py)
#      -> stream_answer(...)          [data_back.py] runs the graph, gives text pieces
#      -> get_pending_approval(...)   [data_back.py] "is a ticket waiting for approval?"
#      -> stream_resume(...)          [data_back.py] continues the graph after Approve/Reject
#      -> save_conversations(...)     [data_back.py] saves chat title
#      -> get_conversations()         [data_back.py] gives the sidebar chat list
#      -> get_conversation_messages() [data_back.py] restores messages from the database
#   The screen is always rebuilt from the database, so it never goes out of sync.

import streamlit as st
import uuid

# CONNECTION: all of these come from data_back.py
from data_back import (
    stream_answer,
    stream_resume,
    get_pending_approval,
    save_conversations,
    get_conversations,
    get_conversation_messages
)

st.title("🤖 martStore — Customer Support Assistant")


# ---------- helper functions ----------

# Reads the saved messages of one chat from the database and keeps only
# the ones a customer should see (tool calls / tool results are hidden)
def load_messages(thread_id: str):
    shown = []

    for msg in get_conversation_messages(thread_id):
        if msg.type == "human":
            shown.append({"role": "user", "content": msg.content})

        elif msg.type == "ai" and msg.content and not msg.tool_calls:
            shown.append({"role": "assistant", "content": msg.content})

    return shown


# Shows the answer word by word while it is being created
def show_stream(stream):
    placeholder = st.empty()
    text = ""

    for piece in stream:
        text += piece
        placeholder.markdown(text + "▌")

    placeholder.markdown(text)


# 1. Create a new conversation ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Show chat history
st.sidebar.title("Chat History")

# CONNECTION: get_conversations() -> data_back.py -> checkpoints.sqlite
history = get_conversations()

# 3. Start a new chat
if st.sidebar.button("➕ New Chat"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

for thread_id, title in history:
    if st.sidebar.button(title, key=thread_id):
        st.session_state.thread_id = thread_id

        # CONNECTION: restores this chat from checkpoints.sqlite
        st.session_state.messages = load_messages(thread_id)

        st.rerun()


# 4. Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 5. Human approval (only visible when the graph is paused)
# CONNECTION: get_pending_approval() -> data_back.py -> checkpoint of this thread
pending = get_pending_approval(st.session_state.thread_id)

if pending:
    with st.chat_message("assistant"):
        st.markdown("**🎫 Ticket waiting for your approval**")
        st.write(f"Issue: {pending['issue']}")
        st.write(f"Priority: {pending['priority']}")

        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve", key="approve_ticket")
        reject = col2.button("❌ Reject", key="reject_ticket")

    if approve or reject:
        if approve:
            decision = "approve"
        else:
            decision = "reject"

        with st.chat_message("assistant"):
            try:
                # CONNECTION: stream_resume() -> data_back.py
                # -> continues ticket_approval_node (creates or rejects the ticket)
                show_stream(stream_resume(decision, st.session_state.thread_id))
            except Exception as e:
                st.error(f"Something went wrong: {e}")

        st.session_state.messages = load_messages(st.session_state.thread_id)
        st.rerun()


# 6. Get user's question
# The input is disabled while a ticket is waiting for approval
user_message = st.chat_input(
    "Ask a question...",
    disabled=pending is not None
)

if user_message:

    # Display user's question
    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):
        st.write(user_message)

    # Save the first question as the title
    history = get_conversations()
    saved_ids = [item[0] for item in history] #history me tuple k form me id +chat hn item 0 mean id + title

    if st.session_state.thread_id not in saved_ids:
        # CONNECTION: save_conversations() -> data_back.py -> conversations table
        save_conversations( # do cheze lega and id  and titel db m save kare ag
            st.session_state.thread_id,
            user_message[:40] # start ke irst 40 will be title
        )

    # Get AI response (streamed)
    # CONNECTION: stream_answer() -> data_back.py graph:
    # router_node -> (chatnode | policy_node | order_node | ticket_prepare) -> END
    with st.chat_message("assistant"):
        try:
            show_stream(stream_answer(user_message, st.session_state.thread_id))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    # Rebuild the screen from the database (adds replies that were not streamed,
    # such as policy answers, order tool replies and the ticket notice)
    st.session_state.messages = load_messages(st.session_state.thread_id)

    st.rerun()
