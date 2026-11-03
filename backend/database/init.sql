CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS doc_artifacts (
   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
   announcement_ref VARCHAR(64) UNIQUE NOT NULL,
   title TEXT NOT NULL,
   broadcast_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
   company_name VARCHAR(255) NOT NULL,
   submitted_by VARCHAR(255) NOT NULL,
   designation VARCHAR(255) NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
   id BIGSERIAL PRIMARY KEY,
   chunk TEXT NOT NULL,
   embedding vector(384) NOT NULL,
   artifact_id UUID NOT NULL REFERENCES doc_artifacts(id) ON DELETE CASCADE,
   tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk)) STORED -- normalise the chunk text for comparison later 
);

-- use hnsw index for cosine similarity search on the embedding vector
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m=16, ef_construction=64);

-- gin index so postgres can quickly search the tsvector column for relevant chunks based on a query
-- does not have to check every chunk
CREATE INDEX IF NOT EXISTS idx_documents_tsv
ON documents
USING gin (tsv_content);

-- check for dupes based on the content hash of the document
ALTER TABLE doc_artifacts ADD COLUMN IF NOT EXISTS content_hash CHAR(64);
CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_artifacts_content_hash
ON doc_artifacts (content_hash);