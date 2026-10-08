from datetime import datetime
import os
from dotenv import load_dotenv

from sqlalchemy import MetaData, Table, Column, BigInteger, create_engine, text
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector, HALFVEC
import uuid

load_dotenv()

# connecting SQLAlchemy to your local Docker Postgres
engine = create_engine(f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5433/{os.getenv('POSTGRES_DB')}")

def insert_document_with_hybrid_tokens(
    announcement_ref: str,
    title: str,
    company_name: str,
    broadcast_datetime: datetime,
    submitted_by: str,
    designation: str,
    chunks: list[str],
    embeddings: list
):
    with Session(engine) as session:
        # Step A: Insert Parent Artifact
        artifact_id = str(uuid.uuid4())
        insert_artifact_sql = text("""
            INSERT INTO doc_artifacts (
                id,
                announcement_ref,
                title,
                broadcast_datetime,
                company_name,
                submitted_by,
                designation
            )
            VALUES (
                :id,
                :ref,
                :title,
                :broadcast_datetime,
                :company,
                :submitted_by,
                :designation
            );
            ON CONFLICT (announcement_ref) DO UPDATE 
            SET title = EXCLUDED.title;
        """)
        session.execute(insert_artifact_sql, {
            "id": artifact_id,
            "ref": announcement_ref,
            "title": title,
            "broadcast_datetime": broadcast_datetime,  # Replace with actual broadcast datetime if available
            "company": company_name,
            "submitted_by": submitted_by,  # Replace with actual submitted by if available
            "designation": designation,  # Replace with actual designation if available
        })

        # Step B: Insert Chunks (Embedding + Automatic BM25 tsvector)
        insert_chunk_sql = text("""
            INSERT INTO documents (artifact_id, chunk, embedding)
            VALUES (:artifact_id, :chunk, :embedding::vector);
        """)

        for chunk, emb in zip(chunks, embeddings):
            session.execute(insert_chunk_sql, {
                "artifact_id": artifact_id,
                "chunk": chunk,
                "embedding": str(emb.tolist())
            })
            
        session.commit()
        print(f"Successfully inserted {len(chunks)} chunks for {announcement_ref}.")
