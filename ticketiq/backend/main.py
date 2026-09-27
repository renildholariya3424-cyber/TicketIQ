from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend import rag

app = FastAPI(title="TicketIQ API", description="AI knowledge assistant for support tickets")


class Question(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = Field(4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": rag.count()}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or ""
    data = await file.read()

    if name.lower().endswith(".csv"):
        docs = rag.parse_csv(data)
    elif name.lower().endswith(".txt"):
        docs = rag.parse_text(data, name)
    else:
        raise HTTPException(400, "Only .csv and .txt files are supported.")

    added = rag.add_documents(docs)
    return {"file": name, "tickets": len(docs), "chunks_added": added, "total_chunks": rag.count()}


@app.post("/ask")
def ask(q: Question):
    try:
        return rag.ask(q.question, q.k)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/reset")
def reset():
    rag.reset()
    return {"status": "index cleared"}
