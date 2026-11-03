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
        # Insert Parent Artifact
        insert_artifact_sql = text("""
            INSERT INTO doc_artifacts (
                announcement_ref,
                title,
                broadcast_datetime,
                company_name,
                submitted_by,
                designation
            )
            VALUES (
                :ref,
                :title,
                :broadcast_datetime,
                :company,
                :submitted_by,
                :designation
            )
            ON CONFLICT (announcement_ref) DO UPDATE 
            SET title = EXCLUDED.title
            RETURNING id;
        """)
        
        returned = session.execute(insert_artifact_sql, {
            "ref": announcement_ref,
            "title": title,
            "broadcast_datetime": broadcast_datetime,
            "company": company_name,
            "submitted_by": submitted_by,
            "designation": designation,
        })
        
        returned_id = returned.fetchone()[0]
        
        # Delete old chunks before inserting new ones to avoid duplicates
        delete_old_chunks_sql = text("""
            DELETE FROM documents WHERE artifact_id = :artifact_id;
        """)
        
        session.execute(delete_old_chunks_sql, {"artifact_id": returned_id})
        
        # Insert Chunks
        insert_chunk_sql = text("""
            INSERT INTO documents (artifact_id, chunk, embedding)
            VALUES (:artifact_id, :chunk, CAST(:embedding AS vector));
        """)

        for chunk, emb in zip(chunks, embeddings):
            session.execute(insert_chunk_sql, {
                "artifact_id": returned_id,
                "chunk": chunk,
                "embedding": str(emb.tolist())
            })
            
        session.commit()
        print(f"Successfully inserted {len(chunks)} chunks for {announcement_ref}.")

def find_ref_by_hash(content_hash: str):
    """Return the announcement_ref that already has this file hash, or None."""
    with Session(engine) as session:
        sql = text("SELECT announcement_ref FROM doc_artifacts WHERE content_hash = :hash")
        row = session.execute(sql, {"hash": content_hash}).fetchone()

        # No row means we have never seen this exact file
        if row is None:
            return None
        return row[0]