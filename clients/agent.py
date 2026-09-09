from typing import TypedDict, Optional, List, Dict, Any, AsyncIterator
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from LLM.llm import llm
from helpers.logger import logger
from clients.retriever import retrieve_docs
from clients.qdrant import ensure_collection_exists
from clients.generation import generate_rag_chain, generate_rag_stream, ANSWER_PROMPT as RAG_ANSWER_PROMPT
from clients.text_to_sql import (
    generate_query_node,
    execute_query_node,
    ANSWER_PROMPT as SQL_ANSWER_PROMPT,
    SQLState
)


# -------------------------------------------------------------
# 1. Router Classification Prompt
# -------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = """You are a smart query classifier for a hybrid AI assistant.
Your task is to classify the user question into exactly ONE of three categories:

1. 'sql'
Choose 'sql' if the user asks about database records, counts, statistics, file metadata (e.g. number of uploaded files, file names, file sizes, upload timestamps, message counts).
Examples:
- "How many files have been uploaded?"
- "كم عدد الملفات في النظام؟"
- "What is the largest file uploaded?"
- "ما هي أسماء الملفات وتواريخ رفعها؟"

2. 'rag'
Choose 'rag' if the user asks about the semantic content, knowledge, concepts, summaries, or specific text found INSIDE the uploaded documents/PDFs.
Examples:
- "What is Seif's work experience according to his CV?"
- "ما هي تفاصيل المشروع المذكور في الملف؟"
- "Summarize the uploaded report."
- "What are the key terms in the document?"

3. 'general'
Choose 'general' for greetings, polite conversation, or general questions that do not need documents or database statistics.
Examples:
- "Hello" / "مرحبا"
- "Who are you?" / "من أنت؟"
- "Thank you" / "شكراً"

Respond with ONLY one word: 'sql', 'rag', or 'general'."""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", "Conversation History:\n{chat_history}\n\nUser Question: {question}\nClassification:")
])


async def classify_intent(question: str, chat_history: str = "") -> str:
    """Classify user question into 'sql', 'rag', or 'general'."""
    chain = router_prompt | llm() | StrOutputParser()
    result = await chain.ainvoke({
        "question": question,
        "chat_history": chat_history or "No previous history"
    })
    
    classification = result.strip().lower()
    if "sql" in classification:
        route = "sql"
    elif "rag" in classification:
        route = "rag"
    else:
        route = "general"
        
    logger.info(f"Router classified question '{question}' as route: '{route}'")
    return route


# -------------------------------------------------------------
# 2. LangGraph State Definition
# -------------------------------------------------------------
class UnifiedAgentState(TypedDict):
    question: str
    chat_history: str
    route: str
    docs: Optional[List[Any]]
    sql_query: Optional[str]
    sql_result: Optional[List[Dict[str, Any]]]
    answer: str


# -------------------------------------------------------------
# 3. LangGraph Workflow Nodes
# -------------------------------------------------------------
async def route_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """Determine the routing path for the incoming question."""
    route = await classify_intent(state["question"], state.get("chat_history", ""))
    return {"route": route}


def route_decision(state: UnifiedAgentState) -> str:
    """Conditional edge decision function."""
    return state.get("route", "rag")


async def rag_agent_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """Execute RAG retrieval from Qdrant and generate answer."""
    ensure_collection_exists()
    docs = await retrieve_docs(state["question"])
    logger.info(f"RAG node retrieved {len(docs)} documents.")
    answer = await generate_rag_chain(docs, state["question"], state.get("chat_history", ""))
    return {"docs": docs, "answer": answer}


async def sql_agent_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """Execute Text-to-SQL generation, validation, and execution."""
    sql_state: SQLState = {
        "question": state["question"],
        "query": "",
        "result": None,
        "answer": "",
        "error": None
    }
    
    # 1. Generate SQL
    gen_result = await generate_query_node(sql_state)
    sql_state["query"] = gen_result["query"]
    
    # 2. Execute SQL
    exec_result = await execute_query_node(sql_state)
    sql_state["result"] = exec_result["result"]
    sql_state["error"] = exec_result["error"]
    
    # 3. Generate Answer
    ans_chain = SQL_ANSWER_PROMPT | llm() | StrOutputParser()
    answer = await ans_chain.ainvoke({
        "question": state["question"],
        "query": sql_state["query"],
        "result": sql_state["result"] or sql_state["error"]
    })
    
    return {
        "sql_query": sql_state["query"],
        "sql_result": sql_state["result"],
        "answer": answer.strip()
    }


async def general_agent_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """Handle general conversational queries."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and polite AI assistant. Answer the user conversationally and concisely."),
        ("human", "Previous Conversation:\n{chat_history}\n\nUser: {question}\nAnswer:")
    ])
    chain = prompt | llm() | StrOutputParser()
    answer = await chain.ainvoke({
        "question": state["question"],
        "chat_history": state.get("chat_history", "None")
    })
    return {"answer": answer.strip()}


# -------------------------------------------------------------
# 4. Build and Compile the Unified Graph
# -------------------------------------------------------------
def build_unified_agent_graph():
    graph = StateGraph(UnifiedAgentState)
    
    # Add nodes
    graph.add_node("router", route_node)
    graph.add_node("rag_branch", rag_agent_node)
    graph.add_node("sql_branch", sql_agent_node)
    graph.add_node("general_branch", general_agent_node)
    
    # Start edge
    graph.add_edge(START, "router")
    
    # Conditional edge from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag_branch",
            "sql": "sql_branch",
            "general": "general_branch"
        }
    )
    
    # End edges
    graph.add_edge("rag_branch", END)
    graph.add_edge("sql_branch", END)
    graph.add_edge("general_branch", END)
    
    return graph.compile()


unified_agent_graph = build_unified_agent_graph()


# -------------------------------------------------------------
# 5. Streaming and Non-Streaming Execution Functions
# -------------------------------------------------------------
async def stream_unified_agent(
    question: str,
    chat_history: str = ""
) -> AsyncIterator[tuple[str, str]]:
    """
    Stream agent response.
    Yields (route, token) where route is yielded first or alongside tokens.
    """
    route = await classify_intent(question, chat_history)
    logger.info(f"Starting agent streaming for route '{route}'...")
    
    if route == "rag":
        ensure_collection_exists()
        docs = await retrieve_docs(question)
        async for chunk in generate_rag_stream(docs, question, chat_history):
            yield (route, chunk)
            
    elif route == "sql":
        # Run SQL pipeline
        sql_state: SQLState = {
            "question": question,
            "query": "",
            "result": None,
            "answer": "",
            "error": None
        }
        gen_res = await generate_query_node(sql_state)
        sql_state["query"] = gen_res["query"]
        
        exec_res = await execute_query_node(sql_state)
        sql_state["result"] = exec_res["result"]
        sql_state["error"] = exec_res["error"]
        
        if sql_state["error"]:
            yield (route, f"Database query error: {sql_state['error']}")
            return
            
        chain = SQL_ANSWER_PROMPT | llm() | StrOutputParser()
        async for chunk in chain.astream({
            "question": question,
            "query": sql_state["query"],
            "result": sql_state["result"]
        }):
            yield (route, chunk)
            
    else: # general
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and polite AI assistant. Answer conversationally and concisely."),
            ("human", "Conversation History:\n{chat_history}\n\nUser: {question}\nAnswer:")
        ])
        chain = prompt | llm() | StrOutputParser()
        async for chunk in chain.astream({
            "question": question,
            "chat_history": chat_history or "None"
        }):
            yield (route, chunk)


async def run_unified_agent(question: str, chat_history: str = "") -> Dict[str, Any]:
    """Execute complete unified agent workflow and return full result dictionary."""
    initial_state: UnifiedAgentState = {
        "question": question,
        "chat_history": chat_history,
        "route": "",
        "docs": None,
        "sql_query": None,
        "sql_result": None,
        "answer": ""
    }
    
    final_state = await unified_agent_graph.ainvoke(initial_state)
    return {
        "question": final_state["question"],
        "route": final_state.get("route"),
        "sql_query": final_state.get("sql_query"),
        "sql_result": final_state.get("sql_result"),
        "sources_count": len(final_state.get("docs") or []),
        "answer": final_state.get("answer")
    }
