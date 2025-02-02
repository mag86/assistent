from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from app.config import settings

class LLMService:
    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        # Инициализация локальной модели
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.models["local-llm"] = LlamaCpp(
            model_path=settings.MODEL_PATH,
            n_gpu_layers=settings.N_GPU_LAYERS,
            n_threads=settings.N_THREADS,
            n_ctx=settings.CONTEXT_LENGTH,
            callback_manager=callback_manager,
            verbose=False,
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            # Специфичный для Mistral формат подсказки
            model_kwargs={"stop": ["</s>"]},
        )

    def get_model(self, model_name: str = "local-llm"):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]