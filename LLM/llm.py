from langchain_groq import ChatGroq
from helpers.config import settings
from functools import lru_cache


@lru_cache(maxsize=None)
def llm():
    return ChatGroq(
    model=settings.llm_model,
    groq_api_key=settings.GROQ_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2)

