from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from typing import List
import psycopg2
import os
import joblib
import numpy as np
import docx
from app.extractors import extract_text_from_pdf  # Adjust import as needed

app = FastAPI()

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "Manojgopi@12"
}

SUPPORTED_TYPES = {".pdf", ".doc", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def parse_resume(file_bytes, filename):
    name = os.path.splitext(filename)[0].replace("_", " ").title()
    return {
        "name": name,
        "email": f"{name.lower().replace(' ', '')}@example.com",
        "skills": ["Python", "React"]
    }

def extract_text(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()
    temp_path = f"temp_upload{ext}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    try:
        if ext == ".pdf":
            text = extract_text_from_pdf(temp_path)
        elif ext == ".docx":
            doc = docx.Document(temp_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f2:
                text = f2.read()
        else:
            text = ""
    finally:
        os.remove(temp_path)
    return text

@app.post("/api/resumes/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    print("Upload endpoint called")
    results = []
    failed_details = []

    # Load embedding models once
    tfidf = joblib.load("embeddings/tfidf.joblib")
    svd = joblib.load("embeddings/svd.joblib")

    conn = get_db_conn()
    cursor = conn.cursor()

    for file in files:
        file_result = {"filename": file.filename}
        ext = os.path.splitext(file.filename)[1].lower()

        # Validation: file type
        if ext not in SUPPORTED_TYPES:
            file_result["upload_status"] = "failed"
            file_result["error"] = "Unsupported file format"
            failed_details.append({"file": file.filename, "reason": "Unsupported file type"})
            results.append(file_result)
            continue

        contents = await file.read()
        # Validation: file size
        if len(contents) > MAX_FILE_SIZE:
            file_result["upload_status"] = "failed"
            file_result["error"] = "File size exceeds 10MB"
            failed_details.append({"file": file.filename, "reason": "File size exceeds 10MB"})
            results.append(file_result)
            continue

        try:
            # Save file to pdf_file table
            cursor.execute(
                "INSERT INTO pdf_file (filename, content) VALUES (%s, %s) RETURNING id",
                (file.filename, psycopg2.Binary(contents))
            )
            resume_id = cursor.fetchone()[0]
            parsed_data = parse_resume(contents, file.filename)

            # Extract text for embedding
            text = extract_text(contents, file.filename)
            if text.strip():
                tfidf_vec = tfidf.transform([text])
                svd_vec = svd.transform(tfidf_vec)
                embedding = svd_vec[0].tolist()  # VECTOR(300) expects a list/array

                # Save to resumes_embeddings table
                cursor.execute(
                    """
                    INSERT INTO resume_embeddings (name, resume_text, embedding, file_data, file_name)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        parsed_data["name"],
                        text,
                        embedding,
                        psycopg2.Binary(contents),
                        file.filename
                    )
                )

            file_result.update({
                "id": f"resume_{resume_id}",
                "upload_status": "success",
                "parsed_data": parsed_data
            })
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            conn.rollback()
            file_result["upload_status"] = "failed"
            file_result["error"] = str(e)
            failed_details.append({"file": file.filename, "reason": str(e)})
        results.append(file_result)

    conn.commit()
    cursor.close()
    conn.close()

    uploaded_count = sum(1 for r in results if r.get("upload_status") == "success")
    failed_count = sum(1 for r in results if r.get("upload_status") == "failed")

    # If all failed, return error response
    if uploaded_count == 0:
        return JSONResponse({
            "success": False,
            "message": "Resume upload failed due to validation error",
            "error": {
                "code": "VALIDATION_ERROR",
                "details": failed_details
            }
        })

    # Otherwise, return normal response
    return JSONResponse({
        "success": True,
        "message": "Files uploaded successfully",
        "data": {
            "uploaded_count": uploaded_count,
            "failed_count": failed_count,
            "resumes": results
        }
    })

@app.get("/api/resumes")
def get_resumes(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    conn = get_db_conn()
    cursor = conn.cursor()

    offset = (page - 1) * limit

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM pdf_file")
    total_items = cursor.fetchone()[0]
    total_pages = (total_items + limit - 1) // limit

    # Fetch resumes with pagination
    cursor.execute(
        """
        SELECT id, filename, created_at
        FROM pdf_file
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        (limit, offset)
    )
    rows = cursor.fetchall()

    resumes = []
    for row in rows:
        resume_id, filename, upload_date = row

        # Fetch parsed data from resume_embeddings if available
        cursor.execute(
            """
            SELECT name
            FROM resume_embeddings
            WHERE file_name = %s
            LIMIT 1
            """,
            (filename,)
        )
        parsed = cursor.fetchone()
        if parsed:
            parsed_data = {
                "name": parsed[0],
                "email": f"{parsed[0].lower().replace(' ', '')}@example.com",
                "skills": ["Python", "React"],
                "experience_years": 5  # You can update this logic as needed
            }
        else:
            parsed_data = None

        resumes.append({
            "id": f"resume_{resume_id}",
            "filename": filename,
            "upload_date": upload_date.isoformat() if upload_date else None,
            "parsed_data": parsed_data,
            "match_score": None
        })

    cursor.close()
    conn.close()

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

