from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.utils.vector_store import VectorStore
from app.config import settings

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.prompt = PromptTemplate(
            template="""<s>[INST] You are a helpful AI assistant. Use the following context to answer the question. If the context doesn't contain relevant information, use your knowledge to answer.

Context: {context}

Question: {question} [/INST]""",
            input_variables=["context", "question"]
        )

    async def process_question(self, question: str, model_name: str = "local-llm", temperature: float = 0.7):
        try:
            # Инициализация LLM
            llm = LlamaCpp(
                model_path=settings.MODEL_PATH,
                temperature=temperature,
                max_tokens=2048,
                n_ctx=4096,
                top_p=1,
                verbose=False,
                n_gpu_layers=1,
                n_threads=4,
                stop=["</s>"],
                streaming=False
            )

            # Поиск релевантных документов
            docs = self.vector_store.similarity_search(question)
            context = "\n".join(doc.page_content for doc in docs) if docs else ""

            # Формируем промпт
            formatted_prompt = self.prompt.format(
                context=context,
                question=question
            )

            # Получаем ответ
            response = llm.invoke(formatted_prompt)

            return {
                "answer": response.strip(),
                "sources": [doc.page_content for doc in docs],
                "model_used": model_name
            }

        except Exception as e:
            print(f"Error in process_question: {str(e)}")
            raise Exception(f"Error processing question: {str(e)}")

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)