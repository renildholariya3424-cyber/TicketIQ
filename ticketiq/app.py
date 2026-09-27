import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="TicketIQ", page_icon="🎫")
st.title("🎫 TicketIQ")
st.caption("Ask questions about your support tickets in plain English.")

with st.sidebar:
    st.header("Upload tickets")
    file = st.file_uploader("CSV or TXT file", type=["csv", "txt"])
    if file and st.button("Index file"):
        with st.spinner("Creating embeddings..."):
            r = requests.post(f"{API_URL}/upload", files={"file": (file.name, file.getvalue())})
        if r.ok:
            d = r.json()
            st.success(f"Indexed {d['tickets']} tickets ({d['chunks_added']} chunks).")
        else:
            st.error(r.json().get("detail", "Upload failed"))

    try:
        total = requests.get(f"{API_URL}/health", timeout=5).json()["indexed_chunks"]
        st.metric("Indexed chunks", total)
    except requests.RequestException:
        st.error("Backend not running. Start it with: uvicorn backend.main:app")

    if st.button("Clear index"):
        requests.delete(f"{API_URL}/reset")
        st.rerun()

question = st.text_input("Your question", placeholder="e.g. How was the login OTP issue fixed?")
k = st.slider("Tickets to retrieve", 1, 10, 4)

if st.button("Ask", type="primary") and question:
    with st.spinner("Searching tickets..."):
        r = requests.post(f"{API_URL}/ask", json={"question": question, "k": k})
    if not r.ok:
        st.error(r.json().get("detail", "Something went wrong"))
    else:
        d = r.json()
        st.subheader("Answer")
        st.write(d["answer"])
        st.subheader("Sources")
        for s in d["sources"]:
            with st.expander(f"Ticket {s['ticket_id']}  (distance {s['distance']})"):
                st.text(s["text"])
