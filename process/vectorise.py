from pathlib import Path

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.database.conn import insert_document_with_hybrid_tokens
import json


vectorizer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunker = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",
        "\uff0c",
        "\u3001",
        "\uff0e",
        "\u3002",
        "",
    ],
)


def extract_pdf_text(pdf_path: Path) -> str:
    document_text = ""
    with pdf_path.open("rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            document_text += page_text
    return document_text


def process_pdf_for_storage(
    pdf_path: Path
    ):
    # Extract text from PDF
    document_text = extract_pdf_text(pdf_path)
    chunks = chunker.split_text(document_text)
    #extract metadata from json file with the same name as the pdf file
    metadata_path = pdf_path.with_suffix(".json")
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "title": pdf_path.stem,
            "company_name": "Unknown Company",
            "broadcast_datetime": None,
            "submitted_by": "Unknown Submitter",
            "designation": "Unknown Designation"
        }

    insert_document_with_hybrid_tokens(
        announcement_ref=pdf_path.stem,
        title=metadata["title"],
        company_name=metadata["company_name"],
        broadcast_datetime=metadata["broadcast_datetime"],
        submitted_by=metadata["submitted_by"],
        designation=metadata["designation"],
        chunks=chunks,
        embeddings=[vectorizer.encode(chunk) for chunk in chunks]
    )
    

if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parent / "data"
    for pdf_path in sorted(target_dir.glob("*.pdf")):
        process_pdf_for_storage(pdf_path)
        print(f"Stored {pdf_path.name} into the database.")