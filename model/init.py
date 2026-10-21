from json import tool

from langgraph.graph import StateGraph, START, END
from typing import Any, Dict, List, Optional, TypedDict, Union, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sklearn.metrics.pairwise import cosine_similarity

LLM_MODEL = "qwen2.5:7b"
RERANKER_MODEL = "BAAI/bge-reranker-base"
reranker = ChatOllama(model=RERANKER_MODEL, temperature=0.3, verbose=True)

# ------ State ------
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages] #convo history
    contexts: List[str] # the list of relevant chunks
    
# ------ Tools ------
@tool
def vector_search(query, vector):
    score = cosine_similarity(query, vector)
    return True

@tool
def sql_query():
    return True

@tool
def synthesise():
    return True

tools = [vector_search, sql_query, synthesise]
# ------ LLM ------
llm =  ChatOllama(model=LLM_MODEL, temperature=0, verbose=True)
llm_with_tools = llm.bind_tools(tools)
# ------ Nodes ------
def reranker(state: AgentState):
    # call reranker model to rearrange the list from most to least relevant
    return True

tool_node = ToolNode(tools)