import os
import asyncio
import uuid
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse

from clients.ingestion_RAG import ingestion_file
from clients.qdrant import vector_store, ensure_collection_exists
from clients.generation import format_chat_history
from clients.postgres_schema import save_message, save_uploaded_file, init_db, get_recent_messages
from clients.text_to_sql import run_text_to_sql
from clients.agent import stream_unified_agent, run_unified_agent
from helpers.config import settings
from helpers.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables and verify vector collection on startup
    logger.info("Starting up FastAPI application and initializing database tables...")
    await init_db()
    logger.info("Database initialized successfully.")
    ensure_collection_exists()
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Minimal RAG API",
    description="A simple Retrieval-Augmented Generation API using FastAPI and Qdrant",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/uploadfile/", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Received file upload request: {file.filename}")
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Rejected file {file.filename}: only PDF files are supported.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"File {file.filename} saved to disk ({len(contents)} bytes). Starting chunking...")
        docs = await asyncio.to_thread(ingestion_file, file_path)
        logger.info(f"Extracted {len(docs)} chunks from {file.filename}. Indexing to Qdrant...")

        ensure_collection_exists()
        await vector_store.aadd_documents(docs)
        logger.info(f"Successfully indexed chunks to Qdrant for {file.filename}.")

        # Save uploaded file record in Postgres
        file_record = await save_uploaded_file(
            filename=file.filename,
            file_path=file_path,
            file_size_bytes=len(contents),
            chunks_count=len(docs)
        )
        logger.info(f"Saved file record in Postgres with ID: {file_record.id}")

        return {
            "message": "File processed, indexed in Qdrant, and recorded in Postgres successfully",
            "file_id": str(file_record.id),
            "filename": file.filename,
            "chunks_count": len(docs)
        }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the file: {str(e)}"
        )


@app.post("/ask/")
async def ask(query: str, user_id: Optional[uuid.UUID] = None, stream: bool = True):
    try:
        if not user_id:
            user_id = uuid.uuid4()

        logger.info(f"Processing unified ask request for user {user_id} (stream={stream}): '{query}'")

        # 1. Fetch recent conversation history (last N messages from settings)
        recent_messages = await get_recent_messages(user_id=user_id, limit=settings.MEMORY_LAST_N)
        chat_history_str = format_chat_history(recent_messages)
        logger.info(f"Loaded {len(recent_messages)} previous messages for user {user_id} (limit={settings.MEMORY_LAST_N}).")

        # 2. Save current user question to Postgres
        await save_message(user_id=user_id, message_body=query, role="user")

        # 3. Handle streaming or standard response via Unified Agent
        if stream:
            async def response_streamer():
                full_answer = []
                try:
                    async for route, chunk in stream_unified_agent(query, chat_history=chat_history_str):
                        full_answer.append(chunk)
                        yield chunk
                finally:
                    complete_text = "".join(full_answer).strip()
                    if complete_text:
                        await save_message(user_id=user_id, message_body=complete_text, role="assistant")
                        logger.info(f"Saved complete streamed answer for user {user_id}.")

            return StreamingResponse(
                response_streamer(),
                media_type="text/plain; charset=utf-8",
                headers={"X-User-Id": str(user_id)}
            )
        else:
            result = await run_unified_agent(query, chat_history=chat_history_str)
            logger.info(f"Unified agent completed for user {user_id} via route '{result.get('route')}'.")

            # Save assistant answer to Postgres
            await save_message(user_id=user_id, message_body=result["answer"], role="assistant")

            return {
                "user_id": user_id,
                "question": query,
                "route": result.get("route"),
                "sql_query": result.get("sql_query"),
                "sources_count": result.get("sources_count", 0),
                "answer": result["answer"],
                "history_used_count": len(recent_messages)
            }

    except Exception as e:
        logger.error(f"Error in unified ask endpoint for query '{query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the request: {str(e)}"
        )


@app.post("/ask-sql/", status_code=status.HTTP_200_OK)
async def ask_sql(question: str):
    logger.info(f"Received Text-to-SQL request: {question}")
    try:
        result = await run_text_to_sql(question)
        logger.info(f"Text-to-SQL completed. Generated query: {result.get('sql_query')}")
        return result
    except Exception as e:
        logger.error(f"Error executing Text-to-SQL for question '{question}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing Text-to-SQL: {str(e)}"
        )