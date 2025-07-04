# Resume Tracking System

This project is a Resume Tracking System built with FastAPI and PostgreSQL. It allows you to upload, parse, store, and match resumes, extracting key information such as name, email, skills, and summary/objective from each resume.

---

## Features

- Upload resumes (PDF, DOCX, TXT)
- Extracts name, email, skills (from skills section), and summary/objective (from summary/career objective/profile section)
- Stores resumes and embeddings in PostgreSQL
- List resumes with pagination
- Download single or all resumes (as ZIP)
- Match resumes to job descriptions using embeddings and skill extraction

---

## Prerequisites

- Python 3.8+
- PostgreSQL database
- [pip](https://pip.pypa.io/en/stable/)

---

## Setup

### 1. Clone the repository

```sh
git clone <your-repo-url>
cd resume_tracking_system
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure PostgreSQL

- Create a PostgreSQL database and user as per your environment.
- Update the `DB_CONFIG` in `app/files_upload.py` and `DB_URI` in `app/db.py` if needed:
    ```python
    DB_CONFIG = {
        "host": "10.30.0.102",
        "port": 5432,
        "database": "vectordb",
        "user": "devuser",
        "password": "StrongPassword123"
    }
    ```

- Ensure the following tables exist:

    ```sql
    CREATE TABLE pdf_file (
        id SERIAL PRIMARY KEY,
        filename TEXT,
        content BYTEA,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE resume_embeddings (
        id SERIAL PRIMARY KEY,
        name TEXT,
        resume_text TEXT,
        embedding FLOAT8[],
        file_data BYTEA,
        file_name TEXT
    );
    ```

### 4. Place Embedding Models

Place your trained `tfidf.joblib` and `svd.joblib` files in the `embeddings/` directory.

---

## Running the API

### 1. Start the FastAPI server

```sh
uvicorn app.files_upload:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` makes the API accessible on your network.
- `--port 8000` is the default port.

### 2. Access the API docs

Open your browser and go to:

```
http://<your-server-ip>:8000/docs
```

---

## API Usage

### 1. Upload Resumes

**Endpoint:** `POST /api/resumes/upload`

- Upload one or more files as `multipart/form-data` with the key `files`.
- Supported types: `.pdf`, `.docx`, `.txt`

**Example with curl:**
```sh
curl -F "files=@resume1.pdf" -F "files=@resume2.docx" http://<your-server-ip>:8000/api/resumes/upload
```

---

### 2. List Resumes (Paginated)

**Endpoint:** `GET /api/resumes`

- Query params:
    - `page` (default: 1)
    - `limit` (default: 10, max: 100)

**Example:**
```
http://<your-server-ip>:8000/api/resumes?page=2&limit=10
```

---

### 3. Download a Resume

**Endpoint:** `GET /api/resumes/{resume_id}/download`

- Example:
    ```
    http://<your-server-ip>:8000/api/resumes/resume_1/download
    ```

---

### 4. Download All Resumes as ZIP

**Endpoint:** `POST /api/resumes/download-all`

- Body: JSON with optional `resume_ids` (list of IDs)
- Example:
    ```sh
    curl -X POST -H "Content-Type: application/json" -d '{"resume_ids": ["resume_1", "resume_2"]}' http://<your-server-ip>:8000/api/resumes/download-all
    ```

---

### 5. Match Resumes to Job Description

**Endpoint:** `POST /api/jobs/process-text-and-match`

- Body: JSON with:
    - `job_description` (string)
    - `title` (optional)
    - `resume_ids` (optional list)

---

### 6. Match Resumes to Job Description File

**Endpoint:** `POST /api/jobs/process-file-and-match`

- Form-data:
    - `file` (job description file)
    - `title` (optional)
    - `resume_ids` (optional, JSON string list)

---

## Extraction Logic

- **Name:** Extracted from the first 3 lines of the resume text (not filename), falling back to email prefix if needed.
- **Email:** Extracted from the resume text using regex.
- **Skills:** Extracted only from the "Skills", "Technical Skills", or "Key Skills" section of the resume.
- **Description:** Extracted from the "Summary", "Career Objective", "Professional Summary", "Profile", or "Objective" section, up to the next major section (like "Education").

---

## Security & Production Notes

- Change default credentials and restrict CORS in production.
- Use HTTPS and secure your database.
- For large deployments, use a production server (e.g., Gunicorn + Nginx).

---

## Troubleshooting

- **Connection refused/timeouts:** Make sure you use `--host 0.0.0.0` and open port 8000 in your firewall.
- **Database errors:** Ensure your tables match the schema above.
- **Extraction issues:** Check that resume text is being extracted correctly and that your embedding models are present.

---

## License
