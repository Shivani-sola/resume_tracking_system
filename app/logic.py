# placeholder
from sqlalchemy import text
from app.db import engine
import pandas as pd
from app.extractors import extract_text_from_pdf
from app.vectorizer import load_models
import io
import os

# Path to your trained models (update if needed)
TFIDF_PATH = "embeddings/tfidf.joblib"
SVD_PATH = "embeddings/svd.joblib"

def save_resume(name, text, vec, file_bytes, filename):
    vec_str = '[' + ','.join(map(str, vec)) + ']'
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO resumes_storage (name, resume_text, embedding, file_data, file_name)
            VALUES (:name, :text, :embedding, :file_data, :file_name)
        """), {
            "name": name, "text": text, "embedding": vec_str,
            "file_data": file_bytes, "file_name": filename
        })

def save_job_description(title, description, vec):
    vec_str = '[' + ','.join(map(str, vec)) + ']'
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO job_embeddings (title, description, embedding)
            VALUES (:title, :description, :embedding)
        """), {
            "title": title, "description": description, "embedding": vec_str
        })
    return vec_str

def match_resumes(jd_vec_str, top_k=5):
    query = f"""
        SELECT id, name, resume_text, embedding <=> '{jd_vec_str}' AS similarity
        FROM resumes_storage
        WHERE embedding IS NOT NULL
        ORDER BY similarity ASC
        LIMIT {top_k};
    """
    return pd.read_sql(query, engine)

def save_pdf_file(filename, file_bytes):
    """Save a PDF file to the pdf_file table."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO pdf_file (filename, content)
            VALUES (:filename, :content)
        """), {"filename": filename, "content": file_bytes})

def get_pdf_file(file_id):
    """Retrieve a PDF file by id from the pdf_file table."""
    query = text("SELECT id, filename, content, uploaded_at FROM pdf_file WHERE id = :id")
    with engine.begin() as conn:
        result = conn.execute(query, {"id": file_id}).fetchone()
    return result

def extract_and_store_resume_embeddings():
    """Extract resumes from pdf_file, generate embeddings, and store in resume_embeddings."""
    tfidf, svd = load_models(TFIDF_PATH, SVD_PATH)
    with engine.begin() as conn:
        pdfs = conn.execute(text("SELECT id, filename, content FROM pdf_file")).fetchall()
        for pdf in pdfs:
            pdf_id, filename, content = pdf
            with open(f"temp_{pdf_id}.pdf", "wb") as f:
                f.write(content)
            text_content = extract_text_from_pdf(f"temp_{pdf_id}.pdf")
            vec = svd.transform(tfidf.transform([text_content]))[0]
            vec_str = '[' + ','.join(map(str, vec)) + ']'
            conn.execute(text("""
                INSERT INTO resume_embeddings (pdf_id, filename, embedding)
                VALUES (:pdf_id, :filename, :embedding)
                ON CONFLICT (pdf_id) DO UPDATE SET embedding = :embedding
            """), {"pdf_id": pdf_id, "filename": filename, "embedding": vec_str})
            os.remove(f"temp_{pdf_id}.pdf")

def store_jd_embedding(jd_title, jd_text):
    """Generate and store embedding for a job description."""
    tfidf, svd = load_models(TFIDF_PATH, SVD_PATH)
    vec = svd.transform(tfidf.transform([jd_text]))[0]
    vec_str = '[' + ','.join(map(str, vec)) + ']'
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO job_embeddings (title, description, embedding)
            VALUES (:title, :description, :embedding)
        """), {"title": jd_title, "description": jd_text, "embedding": vec_str})
    return vec

def score_resumes_against_jd(jd_vec, top_k=5):
    """Compute similarity scores between JD embedding and all resume embeddings."""
    import numpy as np
    with engine.begin() as conn:
        resumes = conn.execute(text("SELECT pdf_id, filename, embedding FROM resume_embeddings")).fetchall()
        scores = []
        for resume in resumes:
            pdf_id, filename, emb_str = resume
            emb = np.array(eval(emb_str))
            sim = np.dot(jd_vec, emb) / (np.linalg.norm(jd_vec) * np.linalg.norm(emb))
            scores.append((pdf_id, filename, float(sim)))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]
