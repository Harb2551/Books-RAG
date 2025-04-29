import pytest 
from src.rag_chain import BooksRAGChain
from src.loader import BooksLoader
from src.vector_store import BooksVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever

class TestBooksRAGIntegration:
 @pytest.mark.asyncio
 async def test_full_pipeline(self,sample_books,sample_books_chunks,books_file):
    loader = BooksLoader(books_file)
    documents = loader.document_load()
    assert documents[0].page_content == sample_books
    text_chunks = loader.create_chunks(documents)
    assert len(text_chunks) == 4
    for i,chunk in enumerate(text_chunks):
      assert chunk.page_content == sample_books_chunks[i].page_content
    vector_store = BooksVectorStore()
    assert isinstance(vector_store.embeddings, OpenAIEmbeddings),f"Expected 'OpenAIEmbeddings', but got {type(vector_store.embeddings)}"
    vector_store.create_vector_store(text_chunks)
    assert isinstance(vector_store.vector_store, FAISS), f"Expected 'FAISS' vectorstore, but got {type(vector_store.vector_store)}"
    rag_chain = BooksRAGChain(vector_store) 
    test_question = "Which year was alaska published"
    assert rag_chain.chain is not None, "The RAG chain should not be None"
    response = await rag_chain.query(test_question)
    assert response is not None, "The response should not be None"
    assert "1988" in response

