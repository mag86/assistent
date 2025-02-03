from pydantic import BaseModel
from typing import List, Optional

class Question(BaseModel):
    text: str
    model_name: str = "local-llm"
    temperature: float = 0.7
    need_audio: bool = False
    lang: str = "ru"

class Document(BaseModel):
    content: str
    metadata: dict = {}

class Answer(BaseModel):
    answer: str
    sources: List[str]
    model_used: str
    audio_url: Optional[str] = None
