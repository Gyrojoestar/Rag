import os
from dotenv import load_dotenv

from sqlalchemy import MetaData, Table, Column, BigInteger, create_engine, text
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector, HALFVEC

load_dotenv()

# connecting SQLAlchemy to your local Docker Postgres
engine = create_engine(f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5433/{os.getenv('POSTGRES_DB')}")

# create session and add objects
with Session(engine) as session:
    pass
