from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.services.llm_service import LLMService
from app.utils.vector_store import VectorStore

class RAGService:
    def __init__(self):
        self.llm_service = LLMService()
        self.vector_store = VectorStore()
        
        # Формат подсказки, оптимизированный для Mistral
        self.prompt_template = """<s>[INST] Using the following context, answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question} [/INST]"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    async def process_question(self, question: str, model_name: str = "local-llm", temperature: float = 0.7):
        # Получаем релевантные документы
        relevant_docs = self.vector_store.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Получаем модель
        llm = self.llm_service.get_model(model_name)
        if hasattr(llm, 'temperature'):
            llm.temperature = temperature
        
        # Создаем и запускаем цепочку
        chain = LLMChain(llm=llm, prompt=self.prompt)
        response = await chain.arun(context=context, question=question)
        
        return {
            "answer": response.strip(),
            "sources": [doc.metadata.get("source", "Unknown") for doc in relevant_docs],
            "model_used": model_name
        }