from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from app.models.schemas import Question, Document, Answer
from app.services.rag_service import RAGService
from app.services.audio_service import AudioService
import secrets
from typing import List
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="LLM RAG Service")
rag_service = RAGService()
audio_service = AudioService()
security = HTTPBasic()

# Укажем абсолютный путь к директории с шаблонами
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Добавьте после инициализации app:
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def check_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "secret")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.post("/ask")
async def ask_question(
    request: Request,
    question: Question,
    username: str = Depends(check_credentials)
):
    try:
        # Добавим логирование
        print(f"Processing question: {question.text}")
        
        result = await rag_service.process_question(
            question.text,
            question.model_name,
            question.temperature
        )
        
        print(f"Got result: {result}")  # Логируем результат
        
        # Если требуется аудио ответ
        if question.need_audio:
            try:
                audio_path = audio_service.text_to_speech(result["answer"], question.lang)
                audio_url = f"/audio/{os.path.basename(audio_path)}"
                # Очистка старых файлов
                audio_service.cleanup_old_files()
                
                return templates.TemplateResponse(
                    "audio_player.html",
                    {
                        "request": request,
                        "answer": result["answer"],
                        "audio_url": audio_url
                    }
                )
            except Exception as audio_error:
                print(f"Audio generation error: {str(audio_error)}")
                # Если ошибка в генерации аудио, вернем хотя бы текстовый ответ
                return result
        
        return result
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")  # Логируем ошибку
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Error processing request",
                "type": type(e).__name__
            }
        )

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = os.path.join("data/audio", filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        audio_path, 
        media_type="audio/mpeg",
        headers={"Accept-Ranges": "bytes"}
    )

@app.post("/documents/upload")
async def upload_documents(
    documents: List[Document],
    username: str = Depends(check_credentials)
):
    try:
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        rag_service.vector_store.add_documents(texts, metadatas)
        return {"message": f"Successfully uploaded {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))