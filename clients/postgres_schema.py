import uuid
from sqlalchemy import Column, String, Text, Integer, BigInteger, DateTime, select
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from helpers.config import settings
from helpers.logger import logger

DATABASE_URL = (
    f"postgresql+asyncpg://{settings.username}:{settings.password}@{settings.host}:{settings.port}/{settings.dbname}"
)

engine = create_async_engine(DATABASE_URL, echo=False)

Base = declarative_base()


class Messages(Base):
    __tablename__ = 'Messages'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    role = Column(String, default="user")
    message_body = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UploadedFiles(Base):
    __tablename__ = 'UploadedFiles'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    chunks_count = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


async def init_db():
    logger.info("Checking and creating missing database tables in PostgreSQL...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")


async def save_message(user_id: uuid.UUID, message_body: str, role: str = "user"):
    async with AsyncSessionLocal() as db:
        msg = Messages(user_id=user_id, message_body=message_body, role=role)
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        logger.info(f"Saved {role} message with ID {msg.id} for user {user_id}.")
        return msg


async def get_recent_messages(user_id: uuid.UUID, limit: int = 4):
    """Retrieve the most recent N messages for a user in chronological order."""
    async with AsyncSessionLocal() as db:
        stmt = (
            select(Messages)
            .where(Messages.user_id == user_id)
            .order_by(Messages.created_at.desc())
            .limit(limit)
        )
        res = await db.execute(stmt)
        messages = res.scalars().all()
        # Return oldest to newest for context flow
        return list(reversed(messages))


async def save_uploaded_file(filename: str, file_path: str, file_size_bytes: int, chunks_count: int):
    async with AsyncSessionLocal() as db:
        record = UploadedFiles(
            filename=filename,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            chunks_count=chunks_count
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        logger.info(f"Saved uploaded file record '{filename}' with ID {record.id}.")
        return record

