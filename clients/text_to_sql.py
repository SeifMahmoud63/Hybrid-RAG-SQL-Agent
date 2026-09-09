import re
import datetime
import uuid
from typing import TypedDict, Optional, Any, List, Dict
from sqlalchemy import text
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from LLM.llm import llm
from clients.postgres_schema import engine
from helpers.logger import logger


# -------------------------------------------------------------
# 1. Database Schema Context
# -------------------------------------------------------------
SCHEMA_DESCRIPTION = """
PostgreSQL Database Schema:

Table: "UploadedFiles"
  - id (UUID, Primary Key): Unique identifier of the uploaded file.
  - filename (VARCHAR): Name of the file (e.g., 'report.pdf').
  - file_path (VARCHAR): Local storage path of the file on disk.
  - file_size_bytes (BIGINT): File size in bytes.
  - chunks_count (INTEGER): Number of chunks extracted and indexed.
  - uploaded_at (TIMESTAMP WITH TIME ZONE): Exact timestamp of upload.

Table: "Messages"
  - id (UUID, Primary Key): Unique identifier of the message.
  - user_id (UUID): Identifier of the user asking the question.
  - message_body (TEXT): Text content of the message (question or answer).
"""

# -------------------------------------------------------------
# 2. SQL Generation Prompt
# -------------------------------------------------------------
SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior PostgreSQL database expert.
Given an input user question, generate a syntactically correct and optimal PostgreSQL query.

{schema}

CRITICAL RULES:
1. Always enclose table names in double quotes exactly as written: "UploadedFiles" and "Messages".
2. Only generate SELECT queries. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
3. Only select the columns needed to answer the question.
4. Unless specified otherwise, limit the result to 20 rows (LIMIT 20).
5. You can use standard PostgreSQL functions like COUNT, SUM, AVG, MAX, MIN, date_trunc, etc.
6. Return ONLY the executable SQL query. Do NOT add markdown blocks (no ```sql), no explanation, and no trailing semicolon.
"""),
    ("human", "Question: {question}\nSQL Query:")
])

# -------------------------------------------------------------
# 3. Final Answer Prompt
# -------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and professional data assistant.
Given the user's question, the SQL query executed, and the query results retrieved from the database, provide a clear, natural, and comprehensive response.

GUIDELINES:
- Answer in the same language as the user's question (if asked in Arabic, answer in fluent Arabic; if in English, answer in English).
- If the query result is empty, inform the user clearly that no matching records were found.
- Naturally incorporate relevant counts, sizes (convert bytes to KB/MB if helpful), and timestamps.
- Keep the response professional, friendly, and precise.
"""),
    ("human", """User Question: {question}
SQL Query: {query}
Database Result: {result}

Answer:""")
])


# -------------------------------------------------------------
# 4. LangGraph State Definition
# -------------------------------------------------------------
class SQLState(TypedDict):
    question: str
    query: str
    result: Optional[List[Dict[str, Any]]]
    answer: str
    error: Optional[str]


# -------------------------------------------------------------
# 5. Helper and Safety Functions
# -------------------------------------------------------------
FORBIDDEN_KEYWORDS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bCREATE\b", r"\bREPLACE\b",
    r"\bGRANT\b", r"\bREVOKE\b", r"\bEXEC\b", r"\bEXECUTE\b"
]

def clean_sql_query(raw_query: str) -> str:
    """Clean markdown tags and trailing semicolons from query."""
    cleaned = raw_query.strip()
    # Remove markdown code blocks if present
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip().rstrip(";")
    return cleaned

def is_safe_query(query: str) -> bool:
    """Check if query is safe and read-only."""
    upper_query = query.upper().strip()
    
    # Must start with SELECT or WITH
    if not (upper_query.startswith("SELECT") or upper_query.startswith("WITH")):
        return False
        
    # Block multiple statements in one query
    if ";" in query:
        return False
        
    # Check for forbidden keywords
    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, upper_query):
            return False
            
    return True

def serialize_row_value(val: Any) -> Any:
    """Convert special types like UUID and DateTime into JSON-friendly values."""
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    return val


# -------------------------------------------------------------
# 6. LangGraph Nodes
# -------------------------------------------------------------
async def generate_query_node(state: SQLState) -> Dict[str, Any]:
    """Node 1: Generate SQL query from user question using LLM."""
    logger.info(f"Generating SQL query for question: '{state['question']}'")
    chain = SQL_GENERATION_PROMPT | llm() | StrOutputParser()
    raw_query = await chain.ainvoke({
        "schema": SCHEMA_DESCRIPTION,
        "question": state["question"]
    })
    
    cleaned_query = clean_sql_query(raw_query)
    logger.info(f"Generated SQL query: {cleaned_query}")
    return {"query": cleaned_query}


async def execute_query_node(state: SQLState) -> Dict[str, Any]:
    """Node 2: Validate query and run it on PostgreSQL."""
    query = state.get("query", "")
    
    # 1. Security check
    if not is_safe_query(query):
        logger.warning(f"Security Alert: Blocked unsafe SQL query: '{query}'")
        return {
            "result": None,
            "error": "Security Alert: Only read-only SELECT queries are permitted on this database."
        }

    # 2. Async database execution
    try:
        logger.info(f"Executing SQL query on database: {query}")
        async with engine.connect() as conn:
            db_res = await conn.execute(text(query))
            mappings = db_res.mappings().all()
            
            # Convert rows to a simple list of dicts
            serialized_rows = [
                {k: serialize_row_value(v) for k, v in dict(row).items()}
                for row in mappings
            ]
            logger.info(f"SQL execution returned {len(serialized_rows)} rows.")
            return {"result": serialized_rows, "error": None}
            
    except Exception as e:
        logger.error(f"Database execution error on query '{query}': {str(e)}", exc_info=True)
        return {
            "result": None,
            "error": f"Database execution error: {str(e)}"
        }


async def generate_answer_node(state: SQLState) -> Dict[str, Any]:
    """Node 3: Generate final answer from query results."""
    error = state.get("error")
    
    if error:
        # Return friendly message if there is an error
        logger.warning(f"Returning error message to user: {error}")
        return {
            "answer": f"Sorry, could not retrieve the data due to an error: {error}"
        }
        
    logger.info("Generating final natural language response from query results...")
    chain = ANSWER_PROMPT | llm() | StrOutputParser()
    answer = await chain.ainvoke({
        "question": state["question"],
        "query": state.get("query", ""),
        "result": state.get("result", [])
    })
    logger.info("Final answer generated successfully.")
    
    return {"answer": answer.strip()}


# -------------------------------------------------------------
# 7. Build and Compile Graph
# -------------------------------------------------------------
def build_sql_graph():
    graph = StateGraph(SQLState)
    
    # Add nodes
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("generate_answer", generate_answer_node)
    
    # Connect nodes in sequence
    graph.add_edge(START, "generate_query")
    graph.add_edge("generate_query", "execute_query")
    graph.add_edge("execute_query", "generate_answer")
    graph.add_edge("generate_answer", END)
    
    return graph.compile()


# Compiled runnable graph instance
sql_graph = build_sql_graph()


# -------------------------------------------------------------
# 8. Main Invocation Function
# -------------------------------------------------------------
async def run_text_to_sql(question: str) -> Dict[str, Any]:
    """Main function to run question through LangGraph pipeline."""
    initial_state: SQLState = {
        "question": question,
        "query": "",
        "result": None,
        "answer": "",
        "error": None
    }
    
    final_state = await sql_graph.ainvoke(initial_state)
    return {
        "question": final_state["question"],
        "sql_query": final_state.get("query"),
        "result": final_state.get("result"),
        "answer": final_state.get("answer"),
        "error": final_state.get("error")
    }
