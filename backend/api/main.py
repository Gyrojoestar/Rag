import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend.database.conn import insert_document_with_hybrid_tokens
from model.init import app as rag_graph  # the compiled LangGraph workflow
from model.init import vectorizer  # reuse the already-loaded embedding model
from process.vectorise import PDF_DIRECTORY, chunker, extract_pdf_text

api = FastAPI(title="RAG Announcements API")

# Simple in-memory record of background jobs: {job_id: "queued" / "processing" / "done" / "failed: ..."}
# (Resets when the server restarts. Fine for coursework.)
jobs = {}


# ------------------------------------------------------------
# Part 1: file upload + background processing
# ------------------------------------------------------------
def ingest_pdf(job_id: str, pdf_path: Path, metadata: dict) -> None:
    """Runs in the background: read PDF -> chunk -> embed -> store in Postgres."""
    jobs[job_id] = "processing"

    try:
        # Step 1: get the text out of the PDF
        document_text = extract_pdf_text(pdf_path)
        if not document_text.strip():
            raise ValueError("PDF has no extractable text")

        # Step 2: split the text into chunks
        chunks = chunker.split_text(document_text)

        # Step 3: turn each chunk into a vector
        embeddings = vectorizer.encode(chunks, batch_size=32, normalize_embeddings=True)

        # Step 4: save everything to the database
        insert_document_with_hybrid_tokens(
            announcement_ref=metadata["announcement_ref"],
            title=metadata["title"],
            company_name=metadata["company_name"],
            broadcast_datetime=metadata["broadcast_datetime"],
            submitted_by=metadata["submitted_by"],
            designation=metadata["designation"],
            chunks=chunks,
            embeddings=embeddings,
        )

        jobs[job_id] = "done"

    except Exception as err:
        # Record the error so the client can see what went wrong
        jobs[job_id] = f"failed: {err}"


@api.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    announcement_ref: str = Form(...),
    title: str = Form(...),
    company_name: str = Form(...),
    broadcast_datetime: str = Form(...),  # ISO format, e.g. 2026-10-26T09:30:00+08:00
    submitted_by: str = Form(...),
    designation: str = Form(...),
):
    # Only accept PDFs
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Convert the date string into a datetime object
    try:
        parsed_datetime = datetime.fromisoformat(broadcast_datetime.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="broadcast_datetime must be ISO format")

    # Save the uploaded file to disk (unique prefix avoids name clashes)
    job_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    save_path = PDF_DIRECTORY / f"{job_id}_{safe_name}"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    file_bytes = await file.read()
    save_path.write_bytes(file_bytes)

    # Bundle the metadata into one dictionary
    metadata = {
        "announcement_ref": announcement_ref,
        "title": title,
        "company_name": company_name,
        "broadcast_datetime": parsed_datetime,
        "submitted_by": submitted_by,
        "designation": designation,
    }

    # Schedule the slow work to run AFTER we send the response
    jobs[job_id] = "queued"
    background_tasks.add_task(ingest_pdf, job_id, save_path, metadata)

    return {"job_id": job_id, "status": "queued"}


@api.get("/upload/{job_id}")
async def check_upload_status(job_id: str):
    status = jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, "status": status}


# ------------------------------------------------------------
# Part 2: streaming chat with Server-Sent Events (SSE)
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


async def sse_token_generator(question: str):
    """Runs the LangGraph workflow and yields each LLM token as an SSE message."""
    inputs = {"messages": [HumanMessage(content=question)]}

    # recursion_limit stops the groundedness -> router loop from running forever
    config = {"recursion_limit": 15}

    try:
        # stream_mode="messages" gives us (token_chunk, metadata) as the LLM generates
        async for message_chunk, metadata in rag_graph.astream(
            inputs, config=config, stream_mode="messages"
        ):
            # Only stream the final answer, not the router or the YES/NO judge
            node_name = metadata.get("langgraph_node")
            if node_name != "synthesis":
                continue

            token = message_chunk.content
            if not token:
                continue

            # SSE format is "data: <text>\n\n". JSON keeps newlines in tokens safe.
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"

    except Exception as err:
        error_payload = json.dumps({"error": str(err)})
        yield f"data: {error_payload}\n\n"

    # Tell the client we are finished
    yield "data: [DONE]\n\n"


@api.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        sse_token_generator(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops proxies (e.g. nginx) buffering the stream
        },
    )