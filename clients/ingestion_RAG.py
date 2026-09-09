from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_voyageai import VoyageAIEmbeddings

from helpers.config import settings
from helpers.logger import logger


def loader(file_path: str):
    logger.info(f"Loading document with PyPDFLoader: {file_path}")
    pdf_loader = PyPDFLoader(file_path)
    pages = pdf_loader.load()
    logger.info(f"Successfully loaded {len(pages)} pages from {file_path}")
    return pages


def ingestion_file(file: str):
    pages = loader(file)
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap
    logger.info(f"Splitting {len(pages)} pages into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    result = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(pages)
    logger.info(f"Created {len(result)} chunks from {file}")
    return result

def embedding_model():
    model_name=settings.EMBEDDING_MODEL
    embeddings_model = VoyageAIEmbeddings(
        model=model_name,
        voyage_api_key=settings.VOYAGE_API_KEY,
    )
    return embeddings_model


    


    
    
    