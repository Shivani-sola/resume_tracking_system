# placeholder
from sqlalchemy import create_engine

DB_URI = "postgresql://devuser:StrongPassword123@10.30.0.102:5432/vectordb"
engine = create_engine(DB_URI)
