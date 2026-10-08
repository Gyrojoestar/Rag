from pathlib import Path

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.database.conn import insert_document_with_hybrid_tokens


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
    pdf_path: Path,
    announcement_ref: str | None = None,
    title: str | None = None,
    company_name: str = "Unknown Entity",
):
    announcement_ref = announcement_ref or pdf_path.stem
    title = title or pdf_path.stem

    document_text = extract_pdf_text(pdf_path)
    chunks = chunker.split_text(document_text)
    if not chunks:
        return

    embeddings = vectorizer.encode(chunks)
    insert_document_with_hybrid_tokens(
        announcement_ref=announcement_ref,
        title=title,
        company_name=company_name,
        chunks=chunks,
        embeddings=embeddings,
    )


if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parent / "data"
    for pdf_path in sorted(target_dir.glob("*.pdf")):
        process_pdf_for_storage(pdf_path)
        print(f"Stored {pdf_path.name} into the database.")