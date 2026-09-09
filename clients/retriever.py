from clients.qdrant import vector_store
from helpers.config import settings
from helpers.logger import logger


def get_retriever():
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K}
    )


retriever = get_retriever()


async def retrieve_docs(query: str):
    logger.info(f"Retrieving relevant documents for query: '{query}'")
    docs = await retriever.ainvoke(query)
    logger.info(f"Retrieved {len(docs)} document chunks.")
    return docs
