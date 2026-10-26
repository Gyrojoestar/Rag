from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database.conn import engine

LLM_MODEL = "qwen2.5:7b"
RERANKER_MODEL = "BAAI/bge-reranker-base"

reranker = CrossEncoder(RERANKER_MODEL)
vectorizer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ------ State ------
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    contexts: List[str]
    reranked_contexts: List[str]
    is_grounded: bool


# ------ Tools ------
@tool
def hybrid_search(query: str, top_k: int = 5) -> str:
    """Searches announcement documents using both semantic (dense) and keyword (sparse) search.

    Args:
        query: The user's question or search phrase.
        top_k: How many chunks to return.
    """
    # Dense search: find chunks whose meaning is closest to the query
    query_vector = vectorizer.encode(query, normalize_embeddings=True).tolist()
    dense_sql = text("""
        SELECT d.id, d.chunk, a.company_name, a.title, a.broadcast_datetime
        FROM documents d
        JOIN doc_artifacts a ON d.artifact_id = a.id
        ORDER BY d.embedding <=> CAST(:emb AS vector)
        LIMIT 20;
    """)

    # Build an OR query from the words in the question
    words = query.split()
    clean_words = []
    for word in words:
        # Keep only letters and numbers so the query syntax can't break
        clean_word = ""
        for character in word:
            if character.isalnum():
                clean_word = clean_word + character
        if clean_word:
            clean_words.append(clean_word)

    or_query = " | ".join(clean_words)

    sparse_sql = text("""
        SELECT d.id, d.chunk, a.company_name, a.title, a.broadcast_datetime
        FROM documents d
        JOIN doc_artifacts a ON d.artifact_id = a.id
        WHERE d.tsv_content @@ to_tsquery('english', :q)
        ORDER BY ts_rank(d.tsv_content, to_tsquery('english', :q)) DESC
        LIMIT 20;
    """)

    with Session(engine) as session:
        dense_rows = session.execute(dense_sql, {"emb": str(query_vector)}).fetchall()
        if or_query:
            sparse_rows = session.execute(sparse_sql, {"q": query}).fetchall()

    # Reciprocal Rank Fusion: combine the two ranked lists into one score per chunk
    scores = {}
    rows_by_id = {}

    for rank, row in enumerate(dense_rows):
        chunk_id = row[0]
        rows_by_id[chunk_id] = row
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank + 1)

    for rank, row in enumerate(sparse_rows):
        chunk_id = row[0]
        rows_by_id[chunk_id] = row
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank + 1)

    # Sort chunk ids by combined score, highest first
    sorted_ids = sorted(scores, key=scores.get, reverse=True)

    # Format the top results the same way as before
    results = []
    for chunk_id in sorted_ids[:top_k]:
        row = rows_by_id[chunk_id]
        formatted_chunk = (
            f"Company: {row[2]}\n"
            f"Title: {row[3]}\n"
            f"Date: {row[4]}\n"
            f"Content: {row[1]}"
        )
        results.append(formatted_chunk)

    if not results:
        return "No relevant document chunks found."

    return "\n\n---\n\n".join(results)


@tool
def sql_query(query_type: str, filter_value: str = "") -> str:
    """Queries structured metadata from the doc_artifacts database.

    Args:
        query_type: One of 'count_by_company', 'recent_announcements', or 'filter_by_submitter'.
        filter_value: Company name, submitter name, or search filter value.
    """
    with Session(engine) as session:
        if query_type == "count_by_company":
            sql = text("""
                SELECT company_name, COUNT(*) as total 
                FROM doc_artifacts 
                WHERE company_name ILIKE :val 
                GROUP BY company_name;
            """)
            rows = session.execute(sql, {"val": f"%{filter_value}%"}).fetchall()
            return "\n".join([f"{row[0]}: {row[1]} announcements" for row in rows]) or "No records found."

        elif query_type == "recent_announcements":
            sql = text("""
                SELECT title, company_name, broadcast_datetime 
                FROM doc_artifacts 
                ORDER BY broadcast_datetime DESC 
                LIMIT 5;
            """)
            rows = session.execute(sql).fetchall()
            return "\n".join([f"- [{row[2]}] {row[1]}: {row[0]}" for row in rows])

        elif query_type == "filter_by_submitter":
            sql = text("""
                SELECT title, company_name, designation 
                FROM doc_artifacts 
                WHERE submitted_by ILIKE :val 
                LIMIT 10;
            """)
            rows = session.execute(sql, {"val": f"%{filter_value}%"}).fetchall()
            return "\n".join([f"- {row[1]} ({row[2]}): {row[0]}" for row in rows]) or "No records found."

        else:
            return "Invalid query_type specified."


tools = [hybrid_search, sql_query]

# ------ LLM ------
llm = ChatOllama(model=LLM_MODEL, temperature=0, verbose=True)
llm_with_tools = llm.bind_tools(tools)


# ------ Nodes ------
def router_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def extract_contexts_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    if hasattr(last_message, "content") and last_message.content:
        raw_text = last_message.content
        chunks = raw_text.split("\n\n---\n\n")
        return {"contexts": chunks}
    return {"contexts": []}


def reranker_node(state: AgentState) -> dict:
    raw_contexts = state.get("contexts", [])
    if not raw_contexts:
        return {"reranked_contexts": []}

    # Search backward to find the original HumanMessage query
    user_query = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_query = message.content
            break

    pairs = [[user_query, chunk] for chunk in raw_contexts]
    scores = reranker.predict(pairs)

    chunk_score_pairs = list(zip(raw_contexts, scores))
    chunk_score_pairs.sort(key=lambda item: item[1], reverse=True)

    reranked_chunks = [chunk for chunk, _ in chunk_score_pairs[:3]]
    return {"reranked_contexts": reranked_chunks}


def groundedness_node(state: AgentState) -> dict:
    reranked_contexts = state.get("reranked_contexts", [])

    if not reranked_contexts:
        return {"is_grounded": False}

    user_query = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_query = message.content
            break

    context_block = "\n\n".join(reranked_contexts)

    judge_prompt = SystemMessage(
        content=(
            "You are a strict factual evaluator.\n"
            "Analyze the provided CONTEXT and determine if it contains enough factual evidence "
            "to answer the QUESTION.\n"
            "Respond strictly with 'YES' or 'NO'. Do not explain.\n\n"
            f"QUESTION: {user_query}\n\n"
            f"CONTEXT:\n{context_block}"
        )
    )

    response = llm.invoke([judge_prompt])
    judge_result = response.content.strip().upper()

    return {"is_grounded": "YES" in judge_result}


def synthesis_node(state: AgentState) -> dict:
    reranked_contexts = state.get("reranked_contexts", [])
    if reranked_contexts:
        context_block = "\n\n---\n\n".join(reranked_contexts)
        system_prompt = SystemMessage(
            content=(
                "You are an AI document assistant. Answer the user query strictly using "
                "the provided context below. Do not assume or extrapolate factual details.\n\n"
                f"=== RETRIEVED CONTEXT ===\n{context_block}"
            )
        )
        full_messages = [system_prompt] + state["messages"]
    else:
        system_prompt = SystemMessage(
            content="You are a helpful AI document assistant. Answer the user prompt directly and concisely."
        )
        full_messages = [system_prompt] + state["messages"]

    response = llm.invoke(full_messages)
    return {"messages": [response]}


# ------ Conditional Routing Functions ------
def route_after_llm(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "synthesis"


def route_after_groundedness(state: AgentState) -> str:
    """Routes execution based on the is_grounded state flag."""
    if state.get("is_grounded", False):
        return "synthesis"
    return "router"


# ------ Graph Construction ------
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("extract_contexts", extract_contexts_node)
workflow.add_node("reranker", reranker_node)
workflow.add_node("groundedness", groundedness_node)
workflow.add_node("synthesis", synthesis_node)

# Add Edges
workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_after_llm,
    {"tools": "tools", "synthesis": "synthesis"}
)

workflow.add_edge("tools", "extract_contexts")
workflow.add_edge("extract_contexts", "reranker")
workflow.add_edge("reranker", "groundedness")

workflow.add_conditional_edges(
    "groundedness",
    route_after_groundedness,
    {"synthesis": "synthesis", "router": "router"}
)

workflow.add_edge("synthesis", END)

# Compile Application
app = workflow.compile().with_config({"recursion_limit": 15})