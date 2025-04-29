from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

class BooksVectorStore:
    def __init__(self):
        """Initialize vector store with OpenAI embeddings.
        TODO: Implement initialization
        """

        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create FAISS vector store from documents.
        TODO: Implement this method
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents.
        TODO: Implement this method
        """

        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Create the vector store first.")

        query_embedding = self.embeddings.embed_query(query)
        return self.vector_store.similarity_search_by_vector(query_embedding, k=k)
