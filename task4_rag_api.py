import os
import json
import sqlite3
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Week 4 Task 4 - RAG Assistant API")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="manual_kb")

def index_pdf_manual(pdf_path="data/sample_manual.pdf"):
    if not os.path.exists(pdf_path):
        docs = [
            {"text": "The CloudSync Pro platform supports auto-backup scheduling under Settings > Backup. Supported protocols are SFTP, S3, and GCS.", "page": 3},
            {"text": "To reset administrative 2FA, contact security team with your employee verification token. Never share SMS OTP.", "page": 7},
            {"text": "Refunds are processed within 5-7 business days for annual tier subscriptions cancelled within 30 days.", "page": 12}
        ]
        for i, d in enumerate(docs):
            collection.add(
                documents=[d["text"]],
                metadatas=[{"page": d["page"]}],
                ids=[f"doc_{i}"]
            )
        print("Indexed default mock manual chunks.")
        return

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    splits = splitter.split_documents(pages)

    for i, s in enumerate(splits):
        page_num = s.metadata.get("page", 1)
        collection.add(
            documents=[s.page_content],
            metadatas=[{"page": page_num}],
            ids=[f"chunk_{i}"]
        )
    print(f"Successfully indexed {len(splits)} chunks from {pdf_path}.")

index_pdf_manual()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ClassifyRequest(BaseModel):
    ticket: str

class ChatRequest(BaseModel):
    query: str

@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    results = collection.query(query_texts=[payload.question], n_results=3)
    
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    context_text = "\n\n".join([f"[Source Page {m.get('page')}]: {d}" for d, m in zip(docs, metas)])

    anti_hallucination_prompt = f"""
You are a factual support assistant. Answer the question ONLY using the provided manual context below.
If the answer cannot be found in the context, reply strictly with: "I don't know."
Do not make assumptions, invent features, or extrapolate.

Context:
{context_text}

Question: {payload.question}
Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=anti_hallucination_prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )

    ans = response.text.strip()
    sources = []
    if "i don't know" not in ans.lower():
        sources = [{"page": m.get("page")} for m in metas]

    return AskResponse(answer=ans, sources=sources)

@app.post("/classify")
def classify_ticket(payload: ClassifyRequest):
    with open("prompts/classifier_v2.txt", "r", encoding="utf-8") as f:
        template = f.read()
    
    prompt = template.format(ticket=payload.ticket)
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return {"category": res.text.strip()}


@app.post("/chat")
def chat_assistant(payload: ChatRequest):
    routing_prompt = f"""
Determine whether this query requires:
1. "rag" - Questions about manual, policies, technical guides, refund terms.
2. "tool" - Queries looking up specific order status or customer details.

Query: "{payload.query}"
Output only "rag" or "tool".
"""
    res = client.models.generate_content(model="gemini-2.5-flash", contents=routing_prompt)
    route = res.text.strip().lower()

    if "rag" in route:
        return ask_question(AskRequest(question=payload.query))
    else:
        return {"type": "tool_executed", "result": "Customer/Order lookup completed."}