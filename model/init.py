from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool  # Fixed: imported from langchain_core
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
def vector_search(query: str, top_k: int = 5) -> str:
    query_vector = vectorizer.encode(query, normalize_embeddings=True).tolist()
    query_vector_str = str(query_vector)

    search_sql = text("""
        SELECT 
            d.chunk,
            a.company_name,
            a.title,
            a.broadcast_datetime,
            1 - (d.embedding <=> CAST(:query_embedding AS vector)) AS similarity_score
        FROM documents d
        JOIN doc_artifacts a ON d.artifact_id = a.id
        ORDER BY d.embedding <=> CAST(:query_embedding AS vector) ASC
        LIMIT :top_k;
    """)

    results = []
    with Session(engine) as session:
        rows = session.execute(
            search_sql,
            {"query_embedding": query_vector_str, "top_k": top_k}
        ).fetchall()

        for row in rows:
            chunk_text = row[0]
            company_name = row[1]
            title = row[2]
            broadcast_datetime = row[3]
            score = row[4]

            formatted_chunk = (
                f"Company: {company_name}\n"
                f"Title: {title}\n"
                f"Date: {broadcast_datetime}\n"
                f"Similarity: {score:.4f}\n"
                f"Content: {chunk_text}"
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


tools = [vector_search, sql_query]

# ------ LLM ------
llm = ChatOllama(model=LLM_MODEL, temperature=0, verbose=True)
llm_with_tools = llm.bind_tools(tools)


# ------ Nodes ------
def router_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def extract_contexts_node(state: AgentState) -> dict:
    """Extracts raw text chunks from the tool execution output into state['contexts']."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "content") and last_message.content:
        raw_text = last_message.content
        chunks = raw_text.split("\n\n---\n\n")
        return {"contexts": chunks}
    return {"contexts": []}


def reranker_node(state: AgentState) -> dict:
    """Reranks retrieved contexts using BAAI/bge-reranker-base."""
    raw_contexts = state.get("contexts", [])
    if not raw_contexts:
        return {"reranked_contexts": []}
    
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
    """Evaluates if the reranked contexts contain sufficient facts to answer the question."""
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
    """Routes to tools if LLM generated a tool call, otherwise to synthesis."""
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

workflow.add_node("router", router_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("extract_contexts", extract_contexts_node)
workflow.add_node("reranker", reranker_node)
workflow.add_node("groundedness", groundedness_node)
workflow.add_node("synthesis", synthesis_node)

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

app = workflow.compile()