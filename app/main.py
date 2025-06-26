# Resume Matching System with PDF/DOCX File & Embedding Storage

from fastapi import FastAPI, UploadFile, File, Form, Query
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pdfplumber
from docx import Document as DocxDoc
import os
import tempfile

# --------------------
# Configuration
# --------------------
DB_URI = "postgresql://user:password@host:port/database"
engine = create_engine(DB_URI)

app = FastAPI(title="Resume Matching API")

# --------------------
# Load Models
# --------------------
tfidf = joblib.load("tfidf_model.pkl")
svd = joblib.load("svd_model.pkl")

# --------------------
# Helpers for Text Extraction
# --------------------
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(docx_path):
    doc = DocxDoc(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# --------------------
# Save Resume File and Embedding
# --------------------
def save_resume(name: str, text: str, file_bytes: bytes, filename: str):
    vec = svd.transform(tfidf.transform([text]))[0]
    vec_str = '[' + ','.join(map(str, vec)) + ']'
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO resumes_storage (name, resume_text, embedding, file_data, file_name)
            VALUES (:name, :text, :embedding, :file_data, :file_name)
            """),
            {
                "name": name,
                "text": text,
                "embedding": vec_str,
                "file_data": file_bytes,
                "file_name": filename
            }
        )

# --------------------
# Save JD and Embedding in Separate Table
# --------------------
def save_job_description(title: str, description: str):
    vec = svd.transform(tfidf.transform([description]))[0]
    vec_str = '[' + ','.join(map(str, vec)) + ']'
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO job_embeddings (title, description, embedding)
            VALUES (:title, :description, :embedding)
            """),
            {"title": title, "description": description, "embedding": vec_str}
        )
    return vec_str

# --------------------
# Find Matching Resumes
# --------------------
def match_resumes(jd_vec_str, top_k=5):
    query = f'''
        SELECT id, name, resume_text, embedding <=> '{jd_vec_str}' AS similarity
        FROM resumes_storage
        WHERE embedding IS NOT NULL
        ORDER BY similarity ASC
        LIMIT {top_k};
    '''
    return pd.read_sql(query, engine)

# --------------------
# Upload Resume Endpoint
# --------------------
@app.post("/upload_resume")
def upload_resume(name: str = Form(...), file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        file_bytes = f.read()

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(tmp_path)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(tmp_path)
    else:
        os.remove(tmp_path)
        return {"error": "Unsupported file format. Please upload PDF or DOCX."}

    os.remove(tmp_path)
    save_resume(name, text, file_bytes, file.filename)
    return {"message": "Resume uploaded, embedded, and stored successfully."}

# --------------------
# Match Endpoint
# --------------------
class JDInput(BaseModel):
    title: str
    text: str

@app.post("/match")
def find_matches(data: JDInput, top_k: int = Query(5)):
    jd_vec_str = save_job_description(data.title, data.text)
    results = match_resumes(jd_vec_str, top_k)
    return results[['id', 'name', 'similarity']].to_dict(orient="records")
