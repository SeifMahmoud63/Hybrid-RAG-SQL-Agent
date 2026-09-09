from typing import List, AsyncIterator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLM.llm import llm
from helpers.logger import logger

RAG_PROMPT_TEMPLATE = """You are a helpful assistant for question-answering tasks.
Use the following pieces of retrieved context and previous conversation history to answer the user's question accurately.
If the answer cannot be found in the context or chat history, say that you don't know.

Context:
{context}

Previous Conversation History:
{chat_history}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def format_chat_history(messages: list) -> str:
    """Format previous messages into a clean dialogue history string."""
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages:
        sender = "User" if getattr(msg, "role", "user") == "user" else "Assistant"
        lines.append(f"{sender}: {msg.message_body}")
    return "\n".join(lines)


async def generate_rag_stream(chunks: List[Document], user_question: str, chat_history: str = "") -> AsyncIterator[str]:
    """Stream LLM response chunk by chunk as an async generator."""
    logger.info(f"Streaming LLM response for question '{user_question}' using {len(chunks)} context chunks...")
    context = "\n\n".join(doc.page_content for doc in chunks) if chunks else "No relevant context found."
    chain = prompt | llm() | StrOutputParser()
    
    async for chunk in chain.astream({
        "context": context,
        "chat_history": chat_history,
        "question": user_question
    }):
        yield chunk
        
    logger.info("LLM streaming finished.")


async def generate_rag_chain(chunks: List[Document], user_question: str, chat_history: str = "") -> str:
    """Generate complete LLM response in a single call."""
    logger.info(f"Generating LLM response for question '{user_question}' using {len(chunks)} context chunks...")
    context = "\n\n".join(doc.page_content for doc in chunks) if chunks else "No relevant context found."
    chain = prompt | llm() | StrOutputParser()
    result = await chain.ainvoke({
        "context": context,
        "chat_history": chat_history,
        "question": user_question
    })
    logger.info("LLM response generated successfully.")
    return result
