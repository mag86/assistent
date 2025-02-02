from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    MODEL_PATH: str = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    VECTOR_STORE_PATH: str = "./data/vector_store"
    
    # Настройки для llama.cpp
    N_GPU_LAYERS: int = 1  # Количество слоев для GPU
    N_THREADS: int = 4     # Количество потоков CPU
    CONTEXT_LENGTH: int = 4096  # Максимальная длина контекста
    
    class Config:
        env_file = ".env"

settings = Settings()