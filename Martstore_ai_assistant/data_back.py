# ==========================================
# HOW EVERYTHING IS CONNECTED (BIG PICTURE)
# ==========================================
#
#   data_fron.py (Streamlit frontend)
#      |  stream_answer(...)  -> sends the question into the graph, streams the reply
#      |  stream_resume(...)  -> resumes a paused graph after Approve / Reject
#      |  get_pending_approval(...) -> asks "is the graph waiting for my approval?"
#      |  save_conversations / get_conversations / get_conversation_messages
#      v
#   data_back.py (this file)
#      START -> router_node
#                 |-- "general" -> chatnode        -> llm
#                 |-- "policy"  -> policy_node     -> answer_policy_question() in data_rag.py
#                 |-- "order"   -> order_node  <-> order_tools (lookup_order -> orders.sqlite)
#                 |                  (loop: LLM -> tool -> LLM, max 3 tool steps)
#                 |-- "ticket"  -> ticket_prepare -> ticket_approval (HUMAN APPROVAL) -> tickets table
#      every route -> END
#      SqliteSaver saves the graph state of every thread_id in checkpoints.sqlite
#
#   data_rag.py
#      polices/*.txt -> chunks -> embeddings -> Chroma -> retriever -> llm -> answer + sources


# ==========================================
# 1. IMPORTS
# ==========================================

import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

# CONNECTION: policy_node (section 13) calls this function from data_rag.py
from data_rag import answer_policy_question


# ==========================================
# 2. INITIALIZE LLM
# ==========================================

# The .env file is OUTSIDE this folder (in a parent folder).
# find_dotenv() searches upward from this file until it finds it.
# If your .env is somewhere else, write its full path instead:
# load_dotenv(r"D:\gen ai\langraph\.env")
load_dotenv(find_dotenv())

# Folder of this file, so the database files are always found in the same place
BASE_DIR = Path(__file__).resolve().parent

# CONNECTION: used by router_node, chatnode, ticket_prepare_node
# and (with tools attached) by order_node
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Maximum number of tool calls allowed for ONE customer question
MAX_TOOL_STEPS = 3

# Pattern for an order ID such as MS102
ORDER_ID_PATTERN = re.compile(r"\bMS\d+\b", re.IGNORECASE)


# ==========================================
# 3. GRAPH STATE AND SYSTEM PROMPTS
# ==========================================

# CONNECTION: "messages" = chat history (from MessagesState)
# "route" = written by router_node, read by route_question
# "ticket_issue" / "ticket_priority" = written by ticket_prepare_node,
#                                      read by ticket_approval_node
class SupportState(MessagesState):
    route: str
    ticket_issue: str
    ticket_priority: str


# CONNECTION: used only by chatnode
system_prompt = SystemMessage(
    content=(
        "You are a helpful customer support assistant for martStore. "
        "Answer clearly and politely. Use conversation history "
        "for follow-up questions. Never invent order details "
        "or company policies."
    )
)

# CONNECTION: used only by order_node
order_system_prompt = SystemMessage(
    content=(
        "You are the martStore order assistant. "
        "To answer anything about an order you MUST call the lookup_order tool "
        "with the order ID. Never guess order details. "
        "If the customer did not give an order ID, check the earlier messages; "
        "if there is still none, ask the customer for it (for example MS102). "
        "You can only look up orders. You cannot cancel, change or refund them. "
        "If the customer wants that, say you can only show order details "
        "and offer to create a support ticket."
    )
)


# ==========================================
# 4. DATABASE CONNECTIONS
# ==========================================

# Conversation titles and LangGraph checkpoints
# CONNECTION: used by section 7 (titles) and section 18 (SqliteSaver)
conn = sqlite3.connect(
    str(BASE_DIR / "checkpoints.sqlite"),
    check_same_thread=False
)

# Business records: orders + support tickets
# CONNECTION: used by get_order and create_ticket (section 8)
orders_conn = sqlite3.connect(
    str(BASE_DIR / "orders.sqlite"),
    check_same_thread=False
)


# ==========================================
# 5. CREATE DATABASE TABLES
# ==========================================

conn.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        thread_id TEXT PRIMARY KEY,
        title TEXT
    )
""")
conn.commit()


orders_conn.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_name TEXT NOT NULL,
        item TEXT NOT NULL,
        status TEXT NOT NULL,
        total_amount REAL NOT NULL
    )
""")

# NEW: support tickets are created only after the human approves
orders_conn.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue TEXT NOT NULL,
        priority TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
""")
orders_conn.commit()


# ==========================================
# 6. INSERT FICTIONAL SAMPLE ORDERS
# ==========================================

orders_conn.executemany(
    """
    INSERT OR IGNORE INTO orders
    (order_id, customer_name, item, status, total_amount)
    VALUES (?, ?, ?, ?, ?)
    """,
    [
        ("MS101", "Ali", "Headphones", "delivered", 5000),
        ("MS102", "Sara", "Keyboard", "processing", 3500),
        ("MS103", "Ahmed", "Mouse", "shipped", 1500)
    ]
)

orders_conn.commit()

# DEBUG: shows in the terminal which DB file and which orders are really used
print("Orders DB file:", BASE_DIR / "orders.sqlite")
print("Orders inside DB:", orders_conn.execute("SELECT order_id FROM orders").fetchall())


# ==========================================
# 7. CONVERSATION HISTORY FUNCTIONS
# ==========================================

# CONNECTION: called by data_fron.py when the first question of a new chat is sent
def save_conversations(thread_id: str, title: str):
    conn.execute(
        "INSERT OR IGNORE INTO conversations VALUES (?, ?)",
        (thread_id, title)
    )
    conn.commit()


# CONNECTION: called by data_fron.py to show the chat list in the sidebar
def get_conversations():
    cursor = conn.execute(
        "SELECT thread_id, title FROM conversations ORDER BY rowid DESC"
    )
    return cursor.fetchall()


# ==========================================
# 8. ORDER AND TICKET DATABASE FUNCTIONS
# ==========================================

# CONNECTION: called by the lookup_order tool (section 9), reads orders.sqlite
def get_order(order_id: str):
    cursor = orders_conn.execute(
        """
        SELECT * FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    )

    return cursor.fetchone()


# CONNECTION: called ONLY by ticket_approval_node, and only after Approve.
# Writes a row in the tickets table and returns a ticket ID like TK001
def create_ticket(issue: str, priority: str) -> str:
    cursor = orders_conn.execute(
        """
        INSERT INTO tickets (issue, priority, status, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (issue, priority, "open", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    orders_conn.commit()

    return f"TK{cursor.lastrowid:03d}"


# ==========================================
# 9. TOOLS (used by the order agent)
# ==========================================

# CONNECTION: the LLM can call this tool by itself.
# The docstring is what the LLM reads to understand when to use it.
@tool
def lookup_order(order_id: str) -> str:
    """Look up a martStore order by its order ID (for example MS102).
    Returns the customer, item, status and total of that order."""

    order = get_order(order_id.strip().upper())

    if order is None:
        return f"Order {order_id} was not found."

    return (
        f"Order ID: {order[0]}\n"
        f"Customer: {order[1]}\n"
        f"Item: {order[2]}\n"
        f"Status: {order[3]}\n"
        f"Total: {order[4]:.2f}"
    )


tools = [lookup_order]

# CONNECTION: same llm, but it is now allowed to call the tools above.
# Used only by order_node
llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 10. SMALL HELPER FUNCTIONS
# ==========================================

# Tool calls and tool results are internal. The normal chat nodes
# only need the human messages and the AI text replies.
# CONNECTION: used by chatnode and ticket_prepare_node
def clean_history(messages):
    cleaned = []

    for msg in messages:
        if msg.type == "human":
            cleaned.append(msg)
        elif msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
            cleaned.append(msg)

    return cleaned


# Counts how many tool results exist since the customer's latest question.
# CONNECTION: used by order_node to stop the loop (bounded execution)
def count_tool_steps(messages):
    count = 0

    for msg in reversed(messages):
        if msg.type == "human":
            break
        if msg.type == "tool":
            count += 1

    return count


# ==========================================
# 11. ROUTER NODE
# ==========================================

# CONNECTION: first node of the graph (START -> router).
# Input : state["messages"] (latest user message)
# Output: {"route": "order" | "policy" | "ticket" | "general"}
#         -> route_question (section 16) reads it and picks the next node
def router_node(state: SupportState):
    question = state["messages"][-1].content

    # Rule 1: the customer clearly asks for a ticket
    if re.search(r"\bticket\b", question, re.IGNORECASE):
        print("Selected route: ticket (word 'ticket' found)")
        return {"route": "ticket"}

    # Rule 2: an order ID such as MS102 always means "order".
    # (Before, we depended only on the LLM, and if its output was not exactly
    # "order" the code fell back to "general". That was the bug.)
    if ORDER_ID_PATTERN.search(question):
        print("Selected route: order (found order ID)")
        return {"route": "order"}

    # Rule 3: ask the LLM for everything else
    response = llm.invoke(
    "You are a routing classifier for the martStore customer support assistant.\n\n"

    "Classify the customer's latest message into exactly ONE category:\n"
    "- order\n"
    "- policy\n"
    "- ticket\n"
    "- general\n\n"

    "RULES:\n"
    "1. Choose 'order' when the customer asks about a specific order, "
    "including its status, details, tracking or delivery progress. "
    "Also choose 'order' for 'where is my order' without an order ID.\n"

    "2. Choose 'policy' when the customer asks about general company "
    "rules, including shipping policies, return conditions, refund "
    "eligibility, cancellation rules, or address-change policies, "
    "without asking about a specific identified order.\n"

    "3. Choose 'ticket' when the customer wants to report a problem, "
    "make a complaint, or talk to a human support agent.\n"

    "4. Choose 'general' for general knowledge, greetings, casual "
    "conversation, or questions unrelated to martStore orders and policies.\n\n"

    "Return only one lowercase word: order, policy, ticket, or general.\n\n"
    f"Customer message: {question}"
    )

    # Look for the word inside the reply, so extra text or punctuation
    # from the LLM does not break the route
    text = response.content.strip().lower()
    found = re.search(r"\b(order|policy|ticket|general)\b", text)

    # Handle unexpected LLM output
    route = found.group(1) if found else "general"

    print("Selected route:", route)

    return {"route": route}


# ==========================================
# 12. GENERAL CHAT NODE
# ==========================================

# CONNECTION: runs when route == "general".
# Uses system_prompt + chat history -> llm -> reply saved in messages.
# "config" is passed to the llm so the reply can be streamed to the frontend.
def chatnode(state: SupportState, config: RunnableConfig):
    messages = [system_prompt] + clean_history(state["messages"])

    response = llm.invoke(messages, config)

    return {"messages": [response]}


# ==========================================
# 13. POLICY RAG NODE
# ==========================================

# CONNECTION: runs when route == "policy".
# Flow: question -> answer_policy_question() in data_rag.py
#       -> returns {"answer", "sources"} -> saved as an AI message
def policy_node(state: SupportState):
    question = state["messages"][-1].content

    # Call the separate RAG pipeline
    result = answer_policy_question(question)

    answer = result["answer"]
    sources = result["sources"]

    response = AIMessage(
        content=f"{answer}\n\nSources: {', '.join(sources)}"
    )

    return {"messages": [response]}


# ==========================================
# 14. ORDER AGENT (TOOL CALLING LOOP)
# ==========================================

# CONNECTION: runs when route == "order".
# The loop is:   order_node -> order_tools -> order_node -> ... -> END
#   order_node : the LLM decides "call lookup_order" or "write the final answer"
#   order_tools : ToolNode runs lookup_order and puts the result in messages
# The loop is bounded by MAX_TOOL_STEPS.
def order_node(state: SupportState, config: RunnableConfig):

    # Safety limit: stop if the tool was already used too many times
    if count_tool_steps(state["messages"]) >= MAX_TOOL_STEPS:
        return {"messages": [AIMessage(
            content="Sorry, I could not finish the order lookup. Please try again."
        )]}

    messages = [order_system_prompt] + state["messages"]

    response = llm_with_tools.invoke(messages, config)

    return {"messages": [response]}


# CONNECTION: conditional edge after order_node.
# If the LLM asked for a tool -> go to order_tools, otherwise finish.
def order_next(state: SupportState):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"


# ==========================================
# 15. TICKET NODES (HUMAN IN THE LOOP)
# ==========================================

# STEP A: prepare the ticket. Nothing is saved yet.
# CONNECTION: runs when route == "ticket", then goes to ticket_approval_node
def ticket_prepare_node(state: SupportState):
    last_question = state["messages"][-1].content.lower()

    # The LLM writes a short issue summary from the conversation
    response = llm.invoke(
        [SystemMessage(content=(
            "Write ONE short sentence that describes the customer's problem "
            "for a support ticket. Include the order ID if one was mentioned. "
            "Output only that sentence."
        ))] + clean_history(state["messages"])
    )
    issue = response.content.strip()

    # Business rule in plain Python (not an LLM guess)
    urgent_words = ["urgent", "asap", "damaged", "broken", "wrong item", "not received", "lost"]

    if any(word in last_question for word in urgent_words):
        priority = "high"
    else:
        priority = "normal"

    notice = AIMessage(
        content="I have prepared a support ticket. Please approve or reject it below."
    )

    return {
        "ticket_issue": issue,
        "ticket_priority": priority,
        "messages": [notice]
    }


# STEP B: pause and wait for the human.
# CONNECTION: interrupt() stops the graph here and saves everything in checkpoints.sqlite.
# data_fron.py sees the pause with get_pending_approval() and shows Approve / Reject buttons.
# When a button is clicked, stream_resume() sends "approve" or "reject" back,
# and the value returned by interrupt() below becomes that word.
def ticket_approval_node(state: SupportState):
    decision = interrupt({
        "issue": state["ticket_issue"],
        "priority": state["ticket_priority"]
    })

    if decision == "approve":
        ticket_id = create_ticket(state["ticket_issue"], state["ticket_priority"])

        answer = (
            "✅ Your support ticket has been created.\n\n"
            f"Ticket ID: {ticket_id}\n\n"
            f"Priority: {state['ticket_priority']}\n\n"
            f"Issue: {state['ticket_issue']}"
        )

    else:
        answer = "❌ The ticket was rejected. No ticket was created."

    return {"messages": [AIMessage(content=answer)]}


# ==========================================
# 16. ROUTE DECISION
# ==========================================

# CONNECTION: used by add_conditional_edges (section 17).
# Reads the "route" value written by router_node
def route_question(state: SupportState):
    return state["route"]


# ==========================================
# 17. BUILD LANGGRAPH
# ==========================================

graph = StateGraph(SupportState)

# CONNECTION: node name -> function
graph.add_node("router", router_node)
graph.add_node("chatnode", chatnode)
graph.add_node("policy_node", policy_node)
graph.add_node("order_node", order_node)
graph.add_node("order_tools", ToolNode(tools))
graph.add_node("ticket_prepare", ticket_prepare_node)
graph.add_node("ticket_approval", ticket_approval_node)

graph.add_edge(START, "router")

# CONNECTION: router_node -> route_question -> one of the four workflows
graph.add_conditional_edges(
    "router",
    route_question,
    {
        "general": "chatnode",
        "policy": "policy_node",
        "order": "order_node",
        "ticket": "ticket_prepare"
    }
)

graph.add_edge("chatnode", END)
graph.add_edge("policy_node", END)

# Order workflow: the tool loop
graph.add_conditional_edges(
    "order_node",
    order_next,
    {
        "tools": "order_tools",
        "end": END
    }
)
graph.add_edge("order_tools", "order_node")

# Ticket workflow: prepare -> human approval -> end
graph.add_edge("ticket_prepare", "ticket_approval")
graph.add_edge("ticket_approval", END)


# ==========================================
# 18. ENABLE SQLITE CHECKPOINTING
# ==========================================

# CONNECTION: saves every message and every pause (interrupt) of every
# thread_id in checkpoints.sqlite. This is why old chats can be restored
# and why a waiting approval is not lost if the page reloads.
checkpointer = SqliteSaver(conn)
checkpointer.setup()

workflow = graph.compile(
    checkpointer=checkpointer
)


# ==========================================
# 19. FUNCTIONS USED BY THE FRONTEND (data_fron.py)
# ==========================================

# Restore the saved messages of one chat
# CONNECTION: called by data_fron.py when a chat is clicked in the sidebar
# and after every answer, so the screen always matches the database
def get_conversation_messages(thread_id: str):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = workflow.get_state(config)

    return state.values.get("messages", [])


# Is the graph paused and waiting for a human decision?
# CONNECTION: called by data_fron.py on every page run.
# Returns {"issue": ..., "priority": ...} or None
def get_pending_approval(thread_id: str):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = workflow.get_state(config)

    for task in state.tasks:
        for item in task.interrupts:
            return item.value

    return None


# Only these nodes stream words one by one.
# (router and policy_node are not streamed; their full text appears at the end)
STREAM_NODES = ["chatnode", "order_node"]


# Runs the graph and yields the reply text piece by piece
def _stream_graph(graph_input, thread_id: str):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    for chunk, metadata in workflow.stream(
        graph_input,
        config=config,
        stream_mode="messages"
    ):
        if metadata.get("langgraph_node") not in STREAM_NODES:
            continue

        # Skip empty chunks and chunks that are tool calls
        if getattr(chunk, "tool_call_chunks", None):
            continue

        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content


# CONNECTION: called by data_fron.py when the customer sends a question
def stream_answer(user_text: str, thread_id: str):
    graph_input = {"messages": [("user", user_text)]}

    return _stream_graph(graph_input, thread_id)


# CONNECTION: called by data_fron.py when Approve / Reject is clicked.
# decision = "approve" or "reject"  -> becomes the return value of interrupt()
def stream_resume(decision: str, thread_id: str):
    return _stream_graph(Command(resume=decision), thread_id)

