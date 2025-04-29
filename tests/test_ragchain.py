import asyncio
from unittest.mock import AsyncMock, Mock
import pytest
from src.rag_chain import BooksRAGChain
from langchain_core.documents import Document


class TestBooksRAGChain:
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store"""
        store = Mock()
        store.vector_store = Mock()
        store.vector_store.as_retriever.return_value = Mock()
        store.similarity_search.return_value = [
            Mock(
                page_content="""399029	/m/023mhg	Alaska	James A. Michener	1988	{"/m/03g3w": "History", "/m/02xlf": "Fiction", "/m/0hwxm": "Historical novel"}	 A sweeping description of the formation of the North American continent. The reader follows the development of the Alaskan terrain over millennia. The city of Los Angeles is now some twenty-four hundred miles south of central Alaska, and since it is moving slowly northward as the San Andreas fault slides irresistibly along, the city is destined eventually to become part """,
                metadata={} 
            )
        ]
        return store

    @pytest.fixture
    def rag_chain(self, mock_vector_store):
        """Create RAG chain with mock vector store"""
        return BooksRAGChain(mock_vector_store)

    def test_get_relevant_documents(self,rag_chain):
        relevant_docs = rag_chain.get_relevant_documents("Who is the author of Alaska",k=1)
        assert len(relevant_docs) == 1
        assert "james" in relevant_docs[0].page_content.lower()

    @pytest.mark.asyncio
    async def test_movie_query(self, rag_chain):
        """Test querying about specific movies"""

        response = await rag_chain.query("what was the year when the book alaska was published")
        assert isinstance(response, str)
        assert "Alaska" in response
        assert "1988" in response.lower()
