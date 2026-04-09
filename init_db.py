from __future__ import annotations

from psycopg2.errors import DuplicateObject
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from db import sync_engine

SQL_STATEMENTS = [
    """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
    """,
    """
    CREATE TABLE IF NOT EXISTS papers (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      title TEXT NOT NULL,
      authors TEXT[],
      year INTEGER,
      abstract TEXT,
      source_type VARCHAR(20) NOT NULL DEFAULT 'arxiv',
      arxiv_id VARCHAR(50),
      pdf_url TEXT,
      local_pdf_path TEXT,
      total_pages INTEGER,
      status VARCHAR(20) NOT NULL DEFAULT 'pending',
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_sections (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
      section_name TEXT NOT NULL,
      page_start INTEGER,
      page_end INTEGER,
      section_order INTEGER,
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_chunks (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
      section_id UUID REFERENCES paper_sections(id) ON DELETE SET NULL,
      chunk_index INTEGER NOT NULL,
      chunk_type VARCHAR(30) DEFAULT 'paragraph',
      page_start INTEGER,
      page_end INTEGER,
      content TEXT NOT NULL,
      token_count INTEGER,
      embedding vector(768),
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    """
    DO $$
    BEGIN
      ALTER TABLE paper_chunks
      ADD COLUMN content_tsv tsvector
      GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(content, ''))
      ) STORED;
    EXCEPTION
      WHEN duplicate_column THEN NULL;
    END $$;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_paper_chunks_content_tsv
    ON paper_chunks USING GIN (content_tsv);
    """,
    """
    DELETE FROM paper_chunks a
    USING paper_chunks b
    WHERE a.ctid < b.ctid
    AND a.paper_id = b.paper_id
    AND a.chunk_index = b.chunk_index;
    """,
    """
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_paper_chunk'
      ) THEN
        ALTER TABLE paper_chunks
        ADD CONSTRAINT uq_paper_chunk
        UNIQUE (paper_id, chunk_index);
      END IF;
    END $$;
    """,
    """
    DELETE FROM paper_sections a
    USING paper_sections b
    WHERE a.ctid < b.ctid
    AND a.paper_id = b.paper_id
    AND a.section_name = b.section_name
    AND COALESCE(a.section_order, -1) = COALESCE(b.section_order, -1);
    """,
    """
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_paper_section'
      ) THEN
        ALTER TABLE paper_sections
        ADD CONSTRAINT uq_paper_section
        UNIQUE (paper_id, section_name, section_order);
      END IF;
    END $$;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper_id ON paper_chunks(paper_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_paper_chunks_embedding
      ON paper_chunks USING ivfflat (embedding vector_cosine_ops)
      WITH (lists = 100);
    """,
    """
    CREATE TABLE IF NOT EXISTS figures (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
      figure_number TEXT,
      caption TEXT,
      page_number INTEGER,
      image_path TEXT,
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tables_data (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
      table_number TEXT,
      content_markdown TEXT,
      page_number INTEGER,
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_history (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
      question TEXT NOT NULL,
      answer TEXT NOT NULL,
      retrieved_chunk_ids UUID[],
      created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """,
]


def initialize_database() -> None:
    try:
        with sync_engine.begin() as connection:
            for statement in SQL_STATEMENTS:
                try:
                    connection.execute(text(statement))
                except DBAPIError as exc:
                    if isinstance(exc.orig, DuplicateObject):
                        continue
                    raise
    finally:
        sync_engine.dispose()

    print("Database initialized successfully.")


if __name__ == "__main__":
    initialize_database()
