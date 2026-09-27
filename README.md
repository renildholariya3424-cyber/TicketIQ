# 🎫 TicketIQ – AI Ticket Knowledge Assistant

TicketIQ lets support teams ask questions about past support tickets in plain English.
It uses **Retrieval-Augmented Generation (RAG)**: tickets are embedded into a FAISS vector index,
the most relevant ones are retrieved for each question, and an LLM writes an answer grounded in those tickets.

## Features
- Upload support tickets as **CSV** or **TXT**
- Semantic search with **Sentence Transformers** embeddings + **FAISS**
- AI answers with **LangChain** + OpenAI, citing the ticket IDs used
- Works without an API key too (shows the most relevant tickets)
- **FastAPI** backend with Swagger docs, simple **Streamlit** UI

## How it works
```
Upload tickets → Chunking → Embeddings (all-MiniLM-L6-v2) → FAISS index
Question → Embed → Top-k similar chunks → LangChain prompt → LLM → Answer + sources
```

## Tech Stack
Python · FastAPI · Streamlit · LangChain · FAISS · Sentence Transformers · OpenAI API · Pandas

## Project Structure
```
ticketiq/
├── backend/
│   ├── main.py        # FastAPI endpoints
│   └── rag.py         # parsing, embeddings, FAISS, LLM answer
├── app.py             # Streamlit UI
├── sample_data/
│   └── tickets.csv    # 10 sample support tickets
├── requirements.txt
└── .env.example
```

## Setup
```bash
git clone https://github.com/renildholariya3424-cyber/ticketiq.git
cd ticketiq
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # then add your OpenAI key (optional)
```

## Run
Start the backend (terminal 1):
```bash
uvicorn backend.main:app --reload
```
Start the UI (terminal 2):
```bash
streamlit run app.py
```
Open http://localhost:8501, upload `sample_data/tickets.csv`, and ask something like
*"How was the duplicate email problem fixed?"*

API docs: http://localhost:8000/docs

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status and number of indexed chunks |
| POST | `/upload` | Upload a CSV/TXT file of tickets |
| POST | `/ask` | `{"question": "...", "k": 4}` → answer + sources |
| DELETE | `/reset` | Clear the vector index |

## CSV Format
Any columns work. A `ticket_id` column is used for citations if present, for example:
`ticket_id, subject, description, resolution, category`

## Future Improvements
- PDF and email ticket ingestion
- Metadata filters (category, date)
- Evaluation of retrieval quality

## Author
**Renil Dholariya** – [GitHub](https://github.com/renildholariya3424-cyber) · [LinkedIn](https://linkedin.com/in/renil-dholariya-023806372)
