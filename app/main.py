# Resume Matching System with PDF/DOCX File & Embedding Storage
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Response, status, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text, create_engine
from typing import List, Optional
from app.db import engine
from app.vectorizer import load_models
import os
import io
import pdfplumber
from docx import Document as DocxDoc
import tempfile
import json
import re
import joblib
import spacy

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
DB_URI = "postgresql://postgres:sgowrav%401@localhost:5432/postgres"
engine = create_engine(DB_URI)

# --------------------
# Load Models
# --------------------
tfidf, svd, vectorizer_transform = load_models("embeddings/tfidf.joblib", "embeddings/svd.joblib", n_components=300)

SUPPORTED_TYPES = {".pdf", ".doc", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Load spaCy model once (at module level)
nlp = spacy.load("en_core_web_sm")

# Load a large list of skills from a file
skills_file = os.path.join(os.path.dirname(__file__), "skills_master_list.txt")
with open(skills_file, "r", encoding="utf-8") as f:
    skills_list = [line.strip() for line in f if line.strip()]

# Dummy parser for demonstration
# Replace with actual parsing logic

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else None

def extract_skills(text):
    doc = nlp(text)
    found = set()
    text_lower = text.lower()
    # 1. Try to extract from a 'Skills' section
    import re
    skills_section = None
    # Look for 'Skills' or similar section headers
    section_match = re.search(r'(skills|technical skills|key skills|core skills)\s*[:\-\n]+(.+?)(\n\s*\n|$)', text, re.IGNORECASE | re.DOTALL)
    if section_match:
        skills_section = section_match.group(2)
        # Split by line or comma/semicolon
        possible_skills = re.split(r'[\n,;•\u2022]', skills_section)
        for skill_candidate in possible_skills:
            skill_candidate = skill_candidate.strip()
            for skill in skills_list:
                if skill.lower() == skill_candidate.lower():
                    found.add(skill)
    # 2. Fallback: match skills in the whole text (as before)
    for skill in skills_list:
        pattern = r'\\b' + re.escape(skill.lower()) + r'\\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    # 3. Optionally, add noun chunks that look like skills
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if chunk_text in skills_list:
            found.add(chunk_text)
    return list(found)

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
        print(f"[DEBUG] Processing file: {file.filename}, ext: {ext}")
        if ext not in SUPPORTED_TYPES:
            print(f"[DEBUG] Unsupported file type: {ext}")
            failed.append({
                "filename": file.filename,
                "upload_status": "failed",
                "error": "Unsupported file format"
            })
            continue
        contents = await file.read()
        print(f"[DEBUG] File size: {len(contents)} bytes")
        if len(contents) > MAX_FILE_SIZE:
            print(f"[DEBUG] File too large: {file.filename}")
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
            elif ext == ".txt":
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
            else:
                text_content = ""
            print(f"[DEBUG] Extracted text length: {len(text_content)}")
        except Exception as e:
            print(f"[DEBUG] Text extraction failed: {e}")
            text_content = ""
        os.remove(tmp_path)
        # Generate embedding if text is available
        embedding_str = None
        if text_content.strip():
            try:
                vec = vectorizer_transform([text_content])[0]
                print(f"[DEBUG] Embedding shape: {vec.shape}, sample: {vec[:5]}")
                embedding_str = '[' + ','.join(map(str, vec)) + ']'
            except Exception as e:
                print(f"[DEBUG] Embedding generation failed: {e}")
                embedding_str = None
        else:
            print(f"[DEBUG] No text extracted for embedding.")
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
                        "embedding": embedding_str,
                        "file_data": contents,
                        "file_name": file.filename
                    }
                )
                resume_id = result.scalar()
            print(f"[DEBUG] Inserted into DB with id: {resume_id}")
            parsed_data = parse_resume(contents, file.filename, text_content)
            uploaded.append({
                "id": get_resume_id(resume_id),
                "filename": file.filename,
                "upload_status": "success",
                "parsed_data": parsed_data
            })
        except Exception as e:
            print(f"[DEBUG] DB insert failed: {e}")
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
    import zipfile
    import io
    resume_ids = body.get("resume_ids", None)
    # Get all resumes or filter by IDs
    with engine.begin() as conn:
        if resume_ids:
            db_ids = [int(rid.replace("resume_", "")) for rid in resume_ids]
            rows = conn.execute(text("SELECT file_name, file_data FROM resumes_storage WHERE id = ANY(:ids)"), {"ids": db_ids}).fetchall()
        else:
            rows = conn.execute(text("SELECT file_name, file_data FROM resumes_storage")).fetchall()
    if not rows:
        return JSONResponse({
            "success": False,
            "message": "No resumes found to download",
            "error": {"code": "NO_RESUMES"}
        }, status_code=404)
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_name, file_data in rows:
            zipf.writestr(file_name, file_data)
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=all_resumes.zip"
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
    # Extract requirements from JD using real logic
    required_skills = extract_skills(job_description)
    experience_years = extract_experience_years(job_description)
    education = None  # TODO: implement extraction if needed
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
    jd_vec = vectorizer_transform([job_description])[0]
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
        sim = None
        if emb_str:
            try:
                import numpy as np
                emb = np.array(eval(emb_str))
                sim = float(np.dot(jd_vec, emb) / (np.linalg.norm(jd_vec) * np.linalg.norm(emb)))
                # Scale similarity to 0-100 for frontend
                sim = round(sim * 100, 1)
            except Exception:
                sim = None
        # Skill matching
        matching_skills = list(set(required_skills) & set(parsed_data.get("skills", [])))
        missing_skills = list(set(required_skills) - set(parsed_data.get("skills", [])))
        # Skill match percent
        skills_match = int(100 * len(matching_skills) / len(required_skills)) if required_skills else 0
        # Experience match percent
        experience_match = None
        if experience_years is not None and parsed_data.get("experience_years") is not None:
            exp_ratio = parsed_data["experience_years"] / experience_years
            experience_match = int(100 * min(exp_ratio, 1.0))
        # Overall fit as string
        overall_fit = None
        if sim is not None:
            if sim >= 80:
                overall_fit = "Excellent"
            elif sim >= 60:
                overall_fit = "Good"
            elif sim >= 40:
                overall_fit = "Average"
            else:
                overall_fit = "Poor"
        matched_resumes.append({
            "id": get_resume_id(db_id),
            "filename": file_name,
            "match_score": sim,
            "match_details": {
                "skills_match": skills_match,
                "experience_match": experience_match,
                "overall_fit": overall_fit
            },
            "parsed_data": parsed_data,
            "missing_skills": missing_skills,
            "matching_skills": matching_skills
        })
        if sim is not None:
            scores.append(sim)
    avg_score = float(round(np.mean(scores), 1)) if scores else None
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
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_TYPES:
        return JSONResponse(status_code=400, content={"success": False, "message": "Unsupported file format", "error": {"code": "UNSUPPORTED_FORMAT"}})
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        return JSONResponse(status_code=400, content={"success": False, "message": "File size exceeds limit", "error": {"code": "FILE_TOO_LARGE"}})
    # Extract text from file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        if ext == ".pdf":
            with pdfplumber.open(tmp_path) as pdf:
                job_description = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif ext == ".docx":
            doc = DocxDoc(tmp_path)
            job_description = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                job_description = f.read()
        else:
            job_description = ""
    except Exception:
        job_description = ""
    os.remove(tmp_path)
    job_title = title or os.path.splitext(file.filename)[0].replace("_", " ").title()
    # Extract requirements from JD using real logic
    required_skills = extract_skills(job_description)
    experience_years = extract_experience_years(job_description)
    education = None  # TODO: implement extraction if needed
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
    jd_vec = vectorizer_transform([job_description])[0]
    # Parse resume_ids if provided
    ids = json.loads(resume_ids) if resume_ids else None
    # Get all resumes or filter by IDs
    with engine.begin() as conn:
        if ids:
            db_ids = [int(rid.replace("resume_", "")) for rid in ids]
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
        sim = None
        if emb_str:
            try:
                emb = np.array(eval(emb_str))
                sim = float(np.dot(jd_vec, emb) / (np.linalg.norm(jd_vec) * np.linalg.norm(emb)))
                # Scale similarity to 0-100 for frontend
                sim = round(sim * 100, 1)
            except Exception:
                sim = None
        # Skill matching
        matching_skills = list(set(required_skills) & set(parsed_data.get("skills", [])))
        missing_skills = list(set(required_skills) - set(parsed_data.get("skills", [])))
        # Skill match percent
        skills_match = int(100 * len(matching_skills) / len(required_skills)) if required_skills else 0
        # Experience match percent
        experience_match = None
        if experience_years is not None and parsed_data.get("experience_years") is not None:
            exp_ratio = parsed_data["experience_years"] / experience_years
            experience_match = int(100 * min(exp_ratio, 1.0))
        # Overall fit as string
        overall_fit = None
        if sim is not None:
            if sim >= 80:
                overall_fit = "Excellent"
            elif sim >= 60:
                overall_fit = "Good"
            elif sim >= 40:
                overall_fit = "Average"
            else:
                overall_fit = "Poor"
        matched_resumes.append({
            "id": get_resume_id(db_id),
            "filename": file_name,
            "match_score": sim,
            "match_details": {
                "skills_match": skills_match,
                "experience_match": experience_match,
                "overall_fit": overall_fit
            },
            "parsed_data": parsed_data,
            "missing_skills": missing_skills,
            "matching_skills": matching_skills
        })
        if sim is not None:
            scores.append(sim)
    avg_score = float(round(np.mean(scores), 1)) if scores else None
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
