CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
   id BIGSERIAL PRIMARY KEY,
   chunk TEXT NOT NULL,
   embedding vector(384) NOT NULL,
   artifact_id UUID NOT NULL REFERENCES doc_artifacts(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS doc_artifacts (
   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
   announcement_ref VARCHAR(64) UNIQUE NOT NULL,
   title TEXT NOT NULL,
   broadcast_datetime TIMESTAMP WITH TIME ZONE,
   company_name VARCHAR(255),
   submitted_by VARCHAR(255),
   designation VARCHAR(255)
);
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m=16, ef_construction=64);

CREATE INDEX IF NOT EXISTS idx_documents_tsv
ON documents
USING gin (tsv_content);