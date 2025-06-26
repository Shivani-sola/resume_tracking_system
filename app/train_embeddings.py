import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.vectorizer import train_models
from sqlalchemy import text
from app.db import engine
from app.extractors import extract_text_from_pdf

# Collect all resume texts from your database
with engine.begin() as conn:
    pdfs = conn.execute(text("SELECT id, filename, content FROM pdf_file")).fetchall()

texts = []
for pdf in pdfs:
    pdf_id, filename, content = pdf
    temp_path = f"temp_{pdf_id}.pdf"
    with open(temp_path, "wb") as f:
        f.write(content)
    text_content = extract_text_from_pdf(temp_path)
    texts.append(text_content)
    os.remove(temp_path)

# Train and save models
os.makedirs("embeddings", exist_ok=True)
train_models(texts, "embeddings/tfidf.joblib", "embeddings/svd.joblib")
print("Models trained and saved.")
