from gtts import gTTS
import os
import uuid
from fastapi.responses import FileResponse

class AudioService:
    def __init__(self):
        self.audio_folder = "data/audio"
        os.makedirs(self.audio_folder, exist_ok=True)

    def text_to_speech(self, text: str, lang: str = 'ru') -> str:
        # Генерируем уникальное имя файла
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(self.audio_folder, filename)
        
        # Конвертируем текст в речь
        tts = gTTS(text=text, lang=lang)
        tts.save(filepath)
        
        return filepath

    def cleanup_old_files(self, max_files: int = 100):
        """Удаляет старые аудио файлы, оставляя только последние max_files"""
        files = os.listdir(self.audio_folder)
        if len(files) > max_files:
            files.sort(key=lambda x: os.path.getctime(os.path.join(self.audio_folder, x)))
            for f in files[:-max_files]:
                os.remove(os.path.join(self.audio_folder, f)) 