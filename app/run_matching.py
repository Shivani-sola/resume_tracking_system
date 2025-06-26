import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.logic import extract_and_store_resume_embeddings, store_jd_embedding, score_resumes_against_jd
from app.extractors import extract_text_from_pdf
from sqlalchemy import text
from app.db import engine

# 1. Extract resumes, generate embeddings, and store them
extract_and_store_resume_embeddings()
print("Resume embeddings extracted and stored.")

# 2. Example: Store a job description embedding
jd_title = "Software Engineer"
jd_text = "We are looking for a software engineer with experience in Python, SQL, and machine learning."
jd_vec = store_jd_embedding(jd_title, jd_text)
print("Job description embedding stored.")

# 3. Score resumes against the job description
scores = score_resumes_against_jd(jd_vec)
print("Top resume scores:")

for pdf_id, filename, sim in scores:
    # Fetch PDF content from DB
    with engine.begin() as conn:
        row = conn.execute(text("SELECT content FROM pdf_file WHERE id = :id"), {"id": pdf_id}).fetchone()
    content = row[0] if row else None
    resume_text = ""
    if content:
        temp_path = f"temp_{pdf_id}.pdf"
        with open(temp_path, "wb") as f:
            f.write(content)
        try:
            resume_text = extract_text_from_pdf(temp_path)
        except Exception as e:
            resume_text = f"[Error extracting text: {e}]"
        os.remove(temp_path)
    preview = resume_text.strip().replace("\n", " ")[:200] + ("..." if len(resume_text) > 200 else "")
    print(f"Resume ID: {pdf_id}, Filename: {filename}, Similarity: {sim:.4f}")
    print(f"Content: {preview}\n")
