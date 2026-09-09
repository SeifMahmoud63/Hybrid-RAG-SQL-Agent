from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY:str
    EMBEDDING_MODEL:str
    VOYAGE_API_KEY:str
    EMBEDDING_DIMENSION:int
    chunk_size:int
    chunk_overlap:int
    TOP_K:int

    # postgresql
    host:str
    port:int
    dbname:str
    username:str
    password:str
    MAX_UPLOAD_SIZE_MB:int

    QDRANT_HOST:str
    QDRANT_PORT:int
    llm_model:str
    MEMORY_LAST_N:int = 4

    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()






