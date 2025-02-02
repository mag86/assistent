from fastapi import FastAPI, UploadFile, File, HTTPException
from app.models.schemas import Question, Document, Answer
from app.services.rag_service import RAGService
from typing import List

app = FastAPI(title="LLM RAG Service")
rag_service = RAGService()

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        result = await rag_service.process_question(
            question.text,
            question.model_name,
            question.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_documents(documents: List[Document]):
    try:
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        rag_service.vector_store.add_documents(texts, metadatas)
        return {"message": f"Successfully uploaded {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))