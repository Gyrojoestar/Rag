import json
from datetime import datetime
from pathlib import Path

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.database.conn import insert_document_with_hybrid_tokens

vectorizer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "config" / "blue_chips.json"
PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw_pdfs"

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
    with pdf_path.open("rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)


def load_metadata_index(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Required metadata file does not exist: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    return {
        (PROJECT_ROOT / rec["pdf_path"]).resolve(): rec
        for rec in records
        if "pdf_path" in rec
    }


def validate_and_load_metadata(pdf_path: Path, metadata_by_pdf: dict) -> tuple:
    metadata = metadata_by_pdf.get(pdf_path.resolve())
    if not metadata:
        raise ValueError(f"No metadata record matches PDF: {pdf_path.relative_to(PROJECT_ROOT)}")

    broadcast_datetime = datetime.fromisoformat(
        metadata["broadcast_datetime"].replace("Z", "+00:00")
    )

    return (
        metadata["announcement_ref"],
        metadata["announcement_title"],
        metadata["company_name"],
        broadcast_datetime,
        metadata["submitted_by"],
        metadata["designation"],
    )


def process_pdf_for_storage(pdf_path: Path, metadata_by_pdf: dict) -> None:
    (
        announcement_ref,
        title,
        company_name,
        broadcast_datetime,
        submitted_by,
        designation,
    ) = validate_and_load_metadata(pdf_path, metadata_by_pdf)

    document_text = extract_pdf_text(pdf_path)
    if not document_text.strip():
        raise ValueError(f"CRITICAL ERROR: PDF '{pdf_path.name}' yielded zero extractable text.")

    chunks = chunker.split_text(document_text)
    embeddings = vectorizer.encode(chunks, batch_size=32, normalize_embeddings=True)

    insert_document_with_hybrid_tokens(
        announcement_ref=announcement_ref,
        title=title,
        company_name=company_name,
        broadcast_datetime=broadcast_datetime,
        submitted_by=submitted_by,
        designation=designation,
        chunks=chunks,
        embeddings=embeddings,
    )


if __name__ == "__main__":
    metadata_by_pdf = load_metadata_index(METADATA_PATH)
    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIRECTORY}")
    else:
        print(f"Found {len(pdf_files)} PDFs in {PDF_DIRECTORY}. Starting ingestion...")

    for pdf_path in pdf_files:
        try:
            process_pdf_for_storage(pdf_path, metadata_by_pdf)
            print(f"Successfully stored {pdf_path.name} into PostgreSQL.")
        except (FileNotFoundError, KeyError, ValueError) as err:
            print(f"Skipping {pdf_path.name}: {err}")