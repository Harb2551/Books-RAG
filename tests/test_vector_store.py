import pytest
from langchain_core.documents import Document

from src.vector_store import BooksVectorStore


class TestBooksVectorStore:
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                page_content="""399029	/m/023mhg	Alaska	James A. Michener	1988	{"/m/03g3w": "History", "/m/02xlf": "Fiction", "/m/0hwxm": "Historical novel"}	 A sweeping description of the formation of the North American continent. The reader follows the development of the Alaskan terrain over millennia. The city of Los Angeles is now some twenty-four hundred miles south of central Alaska, and since it is moving slowly northward as the San Andreas fault slides irresistibly along, the city is destined eventually to become part """,
                metadata={},
            ),
            Document(
                page_content="Inception is a sci-fi action movie about entering dreams.",
                metadata={},
            ),
        ]

    def test_create_vector_store(self, sample_documents):
        """Test creating vector store from documents"""
        store = BooksVectorStore()
        store.create_vector_store(sample_documents)
        assert store.vector_store is not None

    def test_similarity_search(self, sample_documents):
        """Test similarity search functionality"""
        store = BooksVectorStore()
        store.create_vector_store(sample_documents)

        results = store.similarity_search("alaska was written by", k=1)
        assert len(results) == 1
        assert "James" in results[0].page_content

    def test_similarity_search_without_initialization(self):
        """Test error handling when searching without initialization"""
        store = BooksVectorStore()
        with pytest.raises(ValueError):
            store.similarity_search("test query")

