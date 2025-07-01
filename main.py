import psycopg2

import os
 
# DB connection info

conn = psycopg2.connect(

    host="localhost",

    port=5432,

    database="postgres",

    user="postgres",

    password="shiva"

)
 
cursor = conn.cursor()
 
# Local folder path with PDFs

pdf_folder = r"C:\Users\solas\OneDrive\Desktop\resume"
 
# Loop through all PDF files
for filename in os.listdir(pdf_folder):


    if filename.lower().endswith(".pdf"):

        filepath = os.path.join(pdf_folder, filename)

        with open(filepath, 'rb') as file:

            binary_data = file.read()

        cursor.execute("""

            INSERT INTO pdf_file (filename, content)

            VALUES (%s, %s)

        """, (filename, psycopg2.Binary(binary_data)))
 
conn.commit()

cursor.close()

conn.close()

print("✅ All PDFs have been uploaded to PostgreSQL.")

