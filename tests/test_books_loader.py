import pytest
from src.loader import BooksLoader
from langchain_community.document_loaders import TextLoader

class TestBooksLoader:
  
  def test_document_load_and_chunks(self,books_file,sample_books,sample_books_chunks):
      loader = BooksLoader(books_file)
      documents = loader.document_load()
      assert documents[0].page_content == sample_books
      text_chunks = loader.create_chunks(documents)
      assert len(text_chunks) == 4
      for i,chunk in enumerate(text_chunks):
        assert chunk.page_content == sample_books_chunks[i].page_content
        
  def test_file_not_found(self):
      loader = BooksLoader("nonexistent.txt")
      with pytest.raises(FileNotFoundError):
          loader.document_load()