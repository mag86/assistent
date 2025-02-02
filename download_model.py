import os
import requests
from huggingface_hub import hf_hub_download

def download_gguf_model():
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # Создаем директорию для моделей, если она не существует
    os.makedirs("models", exist_ok=True)
    
    # Загружаем модель
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=filename,
        local_dir="models"
    )
    
    print(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_gguf_model()