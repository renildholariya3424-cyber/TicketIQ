import io
import os

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_DIR = os.getenv("INDEX_DIR", "vector_store")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_store = None

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a support assistant. Answer the question using ONLY the support tickets below. "
     "If the tickets do not contain the answer, say you could not find it. "
     "Mention the ticket IDs you used.\n\nTickets:\n{context}"),
    ("human", "{question}"),
])


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def get_store():
    global _store
    if _store is None and os.path.exists(INDEX_DIR):
        # Safe here: we only ever load an index this app saved itself.
        _store = FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    return _store


def parse_csv(data: bytes) -> list[Document]:
    df = pd.read_csv(io.BytesIO(data)).fillna("")
    docs = []
    for i, row in df.iterrows():
        ticket_id = str(row.get("ticket_id", f"row-{i + 1}"))
        text = "\n".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
        docs.append(Document(page_content=text, metadata={"ticket_id": ticket_id}))
    return docs


def parse_text(data: bytes, filename: str) -> list[Document]:
    text = data.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"ticket_id": filename})]


def add_documents(docs: list[Document]) -> int:
    global _store
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0
    store = get_store()
    if store is None:
        _store = FAISS.from_documents(chunks, get_embeddings())
    else:
        store.add_documents(chunks)
    _store.save_local(INDEX_DIR)
    return len(chunks)


def count() -> int:
    store = get_store()
    return store.index.ntotal if store else 0


def reset():
    global _store
    _store = None
    if os.path.exists(INDEX_DIR):
        for name in os.listdir(INDEX_DIR):
            os.remove(os.path.join(INDEX_DIR, name))
        os.rmdir(INDEX_DIR)


def generate_answer(question: str, docs: list[Document]) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "No OPENAI_API_KEY set, so no AI answer was generated. The most relevant tickets are shown below."

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def ask(question: str, k: int = 4) -> dict:
    store = get_store()
    if store is None:
        raise ValueError("No tickets indexed yet. Upload a file first.")

    results = store.similarity_search_with_score(question, k=k)
    docs = [doc for doc, _ in results]
    sources = [
        {"ticket_id": doc.metadata.get("ticket_id"), "distance": round(float(score), 3), "text": doc.page_content}
        for doc, score in results
    ]
    return {"answer": generate_answer(question, docs), "sources": sources}
