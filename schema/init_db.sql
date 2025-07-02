CREATE TABLE IF NOT EXISTS resume_embeddings (
    id SERIAL PRIMARY KEY,
    name TEXT,
    resume_text TEXT,
    embedding VECTOR(300),
    file_data BYTEA,
    file_name TEXT
);

CREATE TABLE IF NOT EXISTS job_embeddings (
    id SERIAL PRIMARY KEY,
    title TEXT,
    description TEXT,
    embedding VECTOR(300),
    created_at TIMESTAMP DEFAULT NOW()
);
