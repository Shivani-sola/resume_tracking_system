import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.logic import extract_and_store_resume_embeddings, store_jd_embedding, score_resumes_against_jd

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
    print(f"Resume ID: {pdf_id}, Filename: {filename}, Similarity: {sim:.4f}")
