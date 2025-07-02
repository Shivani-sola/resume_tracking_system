from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import psycopg2
import os

app = FastAPI()

# DB connection info
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
    # Dummy parser for demonstration
    # Replace with actual parsing logic
    name = os.path.splitext(filename)[0].replace("_", " ").title()
    return {
        "name": name,
        "email": f"{name.lower().replace(' ', '')}@example.com",
        "skills": ["Python", "React", "SQL"]
    }

@app.post("/api/resumes/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    print("Upload endpoint called")
    uploaded = []
    failed = 0

    conn = get_db_conn()
    print("DB connection established")
    cursor = conn.cursor()
    print("DB cursor created")

    for idx, file in enumerate(files):
        print(f"Processing file: {file.filename}")
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_TYPES:
            print(f"Unsupported file type: {file.filename}")
            failed += 1
            continue

        contents = await file.read()
        print(f"Read file: {file.filename}, size: {len(contents)} bytes")
        if len(contents) > MAX_FILE_SIZE:
            print(f"File too large: {file.filename}")
            failed += 1
            continue

        try:
            cursor.execute(
                "INSERT INTO pdf_file (filename, content) VALUES (%s, %s) RETURNING id",
                (file.filename, psycopg2.Binary(contents))
            )
            resume_id = cursor.fetchone()[0]
            print(f"Inserted file: {file.filename} with id {resume_id}")
            parsed_data = parse_resume(contents, file.filename)
            uploaded.append({
                "id": f"resume_{resume_id}",
                "filename": file.filename,
                "upload_status": "success",
                "parsed_data": parsed_data
            })
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            conn.rollback()
            failed += 1

    conn.commit()
    cursor.close()
    conn.close()
    print("DB connection closed")

    return JSONResponse({
        "success": True,
        "message": "Files uploaded successfully",
        "data": {
            "uploaded_count": len(uploaded),
            "failed_count": failed,
            "resumes": uploaded
        }
    })

