from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.config import settings

class VectorStore:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=settings.VECTOR_STORE_PATH,
            embedding_function=self.embeddings
        )

    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        self.vector_store.persist()

    def similarity_search(self, query: str, k: int = 4):
        return self.vector_store.similarity_search(query, k=k)