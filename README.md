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

## Example Output

When you run the matching workflow, you will see output like this:

```
Resume embeddings extracted and stored.
Job description embedding stored.
Top resume scores:
Resume ID: 3, Filename: john_doe.pdf, Similarity: 0.8123
Content: Experienced Python developer with 5 years in machine learning and SQL databases.

Resume ID: 7, Filename: jane_smith.pdf, Similarity: 0.7991
Content: Software engineer skilled in Python, data analysis, and cloud technologies.

Resume ID: 2, Filename: alice_resume.pdf, Similarity: 0.7554
Content: Data scientist with expertise in Python, SQL, and statistical modeling.
```

- **Resume ID**: The unique identifier of the resume in the database
- **Filename**: The name of the PDF file
- **Similarity**: The cosine similarity score between the job description and the resume (higher means more relevant)
- **Content**: The extracted text content from the resume PDF (first few lines or a summary)

You can use these scores and content previews to shortlist the best-matching resumes for a given job description.

---

## License
MIT License
