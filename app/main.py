# Resume Matching System with PDF/DOCX File & Embedding Storage
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Response, status, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text, create_engine
from typing import List, Optional
from app.db import engine
import os
import io
import pdfplumber
from docx import Document as DocxDoc
import tempfile
import json
import re
import joblib

app = FastAPI(title="Resume Matching API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Configuration
# --------------------
DB_URI = "postgresql://postgres:sgowrav%401@localhost:5433/postgres"
engine = create_engine(DB_URI)

# --------------------
# Load Models
# --------------------
tfidf = joblib.load("embeddings/tfidf.joblib")
svd = joblib.load("embeddings/svd.joblib")

SUPPORTED_TYPES = {".pdf", ".doc", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Dummy parser for demonstration
# Replace with actual parsing logic

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else None

def extract_skills(text):
    # Example: look for common skills
    skills_list = ["Python", "React", "SQL", "JavaScript", "Django", "Docker", "AWS"]
    found = [skill for skill in skills_list if skill.lower() in text.lower()]
    return found

def extract_experience_years(text):
    match = re.search(r"(\d+)\+?\s*(years|yrs)\s+of\s+experience", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def parse_resume(file_bytes, filename, text=None):
    name = os.path.splitext(filename)[0].replace("_", " ").title()
    if text is None:
        text = ""
    email = extract_email(text)
    skills = extract_skills(text)
    experience_years = extract_experience_years(text)
    return {
        "name": name,
        "email": email or f"{name.lower().replace(' ', '')}@example.com",
        "skills": skills,
        "experience_years": experience_years
    }

def get_resume_id(db_id):
    return f"resume_{db_id}"

def get_job_id(db_id):
    return f"job_{db_id}"

# --------------------
# API Models
# --------------------
class ResumeOut(BaseModel):
    id: int
    name: str
    file_name: str

class JDOut(BaseModel):
    id: int
    title: str
    description: str

class JDInput(BaseModel):
    title: str
    text: str

class MatchResult(BaseModel):
    id: int
    name: str
    similarity: float

# --------------------
# API Endpoints
# --------------------
@app.post("/api/resumes/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    uploaded = []
    failed = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_TYPES:
            failed.append({
                "filename": file.filename,
                "upload_status": "failed",
                "error": "Unsupported file format"
            })
            continue
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            failed.append({
                "filename": file.filename,
                "upload_status": "failed",
                "error": "File size exceeds limit"
            })
            continue
        # Extract text from file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            if ext == ".pdf":
                with pdfplumber.open(tmp_path) as pdf:
                    text_content = "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif ext == ".docx":
                doc = DocxDoc(tmp_path)
                text_content = "\n".join([para.text for para in doc.paragraphs])
            else:
                text_content = ""
        except Exception:
            text_content = ""
        os.remove(tmp_path)
        try:
            name = os.path.splitext(file.filename)[0].replace("_", " ").title()
            with engine.begin() as conn:
                result = conn.execute(
                    text("""
                        INSERT INTO resumes_storage (name, resume_text, embedding, file_data, file_name)
                        VALUES (:name, :resume_text, :embedding, :file_data, :file_name)
                        RETURNING id
                    """),
                    {
                        "name": name,
                        "resume_text": text_content,
                        "embedding": None,
                        "file_data": contents,
                        "file_name": file.filename
                    }
                )
                resume_id = result.scalar()
            parsed_data = parse_resume(contents, file.filename, text_content)
            # Remove experience_years if you want the response to match your example
            parsed_data.pop("experience_years", None)
            uploaded.append({
                "id": get_resume_id(resume_id),
                "filename": file.filename,
                "upload_status": "success",
                "parsed_data": parsed_data
            })
        except Exception as e:
            failed.append({
                "filename": file.filename,
                "upload_status": "failed",
                "error": str(e)
            })
    if not uploaded and failed:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "Resume upload failed due to validation error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "details": [
                        {"file": f["filename"], "reason": f["error"]} for f in failed
                    ]
                }
            }
        )
    return JSONResponse({
        "success": True,
        "message": "Files uploaded successfully",
        "data": {
            "uploaded_count": len(uploaded),
            "failed_count": len(failed),
            "resumes": uploaded + failed
        }
    })

@app.get("/api/resumes")
def get_resumes(page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100)):
    offset = (page - 1) * limit
    with engine.begin() as conn:
        total_items = conn.execute(text("SELECT COUNT(*) FROM resumes_storage")).scalar()
        # Try to fetch uploaded_at if it exists, else fallback to None
        try:
            rows = conn.execute(text("SELECT id, file_name, resume_text, uploaded_at FROM resumes_storage ORDER BY id DESC LIMIT :limit OFFSET :offset"), {"limit": limit, "offset": offset}).fetchall()
            has_uploaded_at = True
        except Exception:
            rows = conn.execute(text("SELECT id, file_name, resume_text FROM resumes_storage ORDER BY id DESC LIMIT :limit OFFSET :offset"), {"limit": limit, "offset": offset}).fetchall()
            has_uploaded_at = False
    resumes = []
    for row in rows:
        if has_uploaded_at:
            db_id, file_name, resume_text, uploaded_at = row
        else:
            db_id, file_name, resume_text = row
            uploaded_at = None
        parsed_data = parse_resume(b"", file_name, resume_text)
        resumes.append({
            "id": get_resume_id(db_id),
            "filename": file_name,
            "upload_date": uploaded_at.isoformat() if uploaded_at else None,
            "parsed_data": parsed_data,
            "match_score": None
        })
    total_pages = (total_items + limit - 1) // limit
    return {
        "success": True,
        "data": {
            "resumes": resumes,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_items": total_items,
                "items_per_page": limit
            }
        }
    }

@app.get("/api/resumes/{resume_id}/download")
def download_resume(resume_id: str, format: str = Query("pdf")):
    try:
        db_id = int(resume_id.replace("resume_", ""))
    except Exception:
        return JSONResponse(status_code=404, content={"success": False, "error": {"code": "NOT_FOUND", "message": "Resume not found"}})
    with engine.begin() as conn:
        row = conn.execute(text("SELECT file_name, file_data FROM resumes_storage WHERE id = :id"), {"id": db_id}).fetchone()
    if not row:
        return JSONResponse(status_code=404, content={"success": False, "error": {"code": "NOT_FOUND", "message": "Resume not found"}})
    file_name, file_data = row
    return StreamingResponse(io.BytesIO(file_data), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=\"{file_name}\""})

@app.post("/api/resumes/download-all")
def download_all_resumes(body: dict = Body(...), format: str = Query("zip")):
    # This is a stub. Implement ZIP creation as needed.
    return JSONResponse({
        "success": False,
        "message": "Not implemented",
        "error": {"code": "NOT_IMPLEMENTED"}
    })

@app.post("/api/resumes/upload-urls")
def upload_resumes_from_urls(body: dict = Body(...)):
    urls = body.get("urls", [])
    processed_count = len(urls)
    successful_uploads = []
    failed_uploads = []
    for idx, url in enumerate(urls):
        # Dummy: Only succeed for the first, fail for the rest
        if idx == 0:
            successful_uploads.append({
                "url": url,
                "id": get_resume_id(idx+1),
                "filename": os.path.basename(url)
            })
        else:
            failed_uploads.append({
                "url": url,
                "error": "File size exceeds limit"
            })
    return JSONResponse({
        "success": True,
        "message": "URLs processed successfully",
        "data": {
            "processed_count": processed_count,
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads
        }
    })

@app.post("/api/jobs/process-text-and-match")
def process_text_and_match(body: dict = Body(...)):
    job_title = body.get("title", "Job")
    job_description = body.get("job_description", "")
    resume_ids = body.get("resume_ids", None)
    # Extract requirements from JD
    required_skills = extract_skills(job_description)
    experience_years = extract_experience_years(job_description)
    education = None  # You can add logic to extract education if needed
    job_analysis = {
        "title": job_title,
        "processed_description": job_description,
        "extracted_requirements": {
            "required_skills": required_skills,
            "preferred_skills": [],
            "experience_years": experience_years,
            "education": education
        },
        "job_category": None,
        "seniority_level": None
    }
    # Generate JD embedding
    jd_vec = svd.transform(tfidf.transform([job_description]))[0]
    # Get all resumes or filter by IDs
    with engine.begin() as conn:
        if resume_ids:
            db_ids = [int(rid.replace("resume_", "")) for rid in resume_ids]
            rows = conn.execute(text("SELECT id, file_name, resume_text, embedding FROM resumes_storage WHERE id = ANY(:ids)"), {"ids": db_ids}).fetchall()
        else:
            rows = conn.execute(text("SELECT id, file_name, resume_text, embedding FROM resumes_storage")).fetchall()
    import numpy as np
    matched_resumes = []
    scores = []
    for row in rows:
        db_id, file_name, resume_text, emb_str = row
        parsed_data = parse_resume(b"", file_name, resume_text)
        # Compute similarity if embedding exists
        if emb_str:
            emb = np.array(eval(emb_str))
            sim = float(np.dot(jd_vec, emb) / (np.linalg.norm(jd_vec) * np.linalg.norm(emb)))
        else:
            sim = None
        # Skill matching
        matching_skills = list(set(required_skills) & set(parsed_data["skills"]))
        missing_skills = list(set(required_skills) - set(parsed_data["skills"]))
        matched_resumes.append({
            "id": get_resume_id(db_id),
            "filename": file_name,
            "match_score": sim,
            "match_details": {
                "skills_match": len(matching_skills),
                "experience_match": parsed_data.get("experience_years"),
                "overall_fit": None
            },
            "parsed_data": parsed_data,
            "missing_skills": missing_skills,
            "matching_skills": matching_skills
        })
        if sim is not None:
            scores.append(sim)
    avg_score = float(np.mean(scores)) if scores else None
    return JSONResponse({
        "success": True,
        "message": "Job description processed and resumes matched successfully",
        "data": {
            "job_analysis": job_analysis,
            "matched_resumes": matched_resumes,
            "total_matched": len(matched_resumes),
            "average_score": avg_score
        }
    })

@app.post("/api/jobs/process-file-and-match")
async def process_file_and_match(file: UploadFile = File(...), title: Optional[str] = Form(None), resume_ids: Optional[str] = Form(None)):
    # Dummy implementation for demonstration
    job_title = title or "Senior Python Developer"
    job_description = "..."
    ids = json.loads(resume_ids) if resume_ids else ["resume_1"]
    job_analysis = {
        "title": job_title,
        "processed_description": job_description,
        "extracted_requirements": {
            "required_skills": ["Python", "Django", "SQL"],
            "preferred_skills": ["Docker", "AWS"],
            "experience_years": 5,
            "education": "Bachelor's degree in Computer Science"
        },
        "job_category": "Software Development",
        "seniority_level": "Senior"
    }
    matched_resumes = [
        {
            "id": rid,
            "filename": "john_doe_resume.pdf",
            "match_score": 85.5,
            "match_details": {
                "skills_match": 90,
                "experience_match": 80,
                "overall_fit": "Excellent"
            },
            "parsed_data": {
                "name": "John Doe",
                "email": "john@example.com",
                "skills": ["Python", "React", "SQL", "JavaScript"],
                "experience_years": 5
            },
            "missing_skills": ["Docker"],
            "matching_skills": ["Python", "Django", "SQL"]
        } for rid in ids
    ]
    return JSONResponse({
        "success": True,
        "message": "Job description processed and resumes matched successfully",
        "data": {
            "job_analysis": job_analysis,
            "matched_resumes": matched_resumes,
            "total_matched": len(matched_resumes),
            "average_score": 85.5
        }
    })
