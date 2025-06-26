Resume Match System
This project is a resume matching system that:

Stores PDF resumes in a PostgreSQL database
Extracts text from resumes and generates embeddings using TF-IDF + SVD
Stores and embeds job descriptions
Computes similarity scores between job descriptions and resumes to find the best matches


Features

Upload and store resumes as PDFs in the database
Extract and embed resume and job description text
Match resumes to job descriptions using cosine similarity
All logic implemented in Python using SQLAlchemy, scikit-learn, and pdfplumber
Setup

Install dependencies:

pip install -r requirements.txt

Configure your PostgreSQL database:

Create the required tables (pdf_file, resume_embeddings, job_embeddings).
Add PDF resumes to the pdf_file table.

Train embedding models:

Run the matching workflow:
   
     python app/run_matching.py

Project Structure
logic.py - Main logic for database operations and matching
extractors.py - PDF and DOCX text extraction
vectorizer.py - Embedding model training and loading
train_embeddings.py - Script to train and save embedding models
run_matching.py - Script to run the end-to-end matching workflow
Requirements
See requirements.txt for all dependencies.
