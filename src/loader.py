from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class BooksLoader:

    def __init__(self,path):
        """Initialise file path."""
        self.filepath = path

    def document_load(self):
        """Load articles from data/books.txt Text file."""
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"File not found: {self.filepath}")
            loader = TextLoader(self.filepath)
            documents = loader.load()
            return documents
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading books: {str(e)}")

    def create_chunks(self,documents):
        """split the documents into chunks of size 500 and overlap of 50"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)
        return text_chunks
