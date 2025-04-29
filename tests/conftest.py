import pytest
import tempfile
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

@pytest.fixture
def sample_books():
  return """399029	/m/023mhg Book Name -	Alaska	James A. Michener Published in	1988	{"/m/03g3w": "History", "/m/02xlf": "Fiction", "/m/0hwxm": "Historical    
           novel"}	 A sweeping description of the formation of the North American continent. The reader follows the development of the Alaskan terrain over millennia. The city of Los Angeles is now some twenty-four hundred miles south of central Alaska, and since it is moving slowly northward as the San Andreas fault slides irresistibly along, the city is destined eventually to become part 
           400425	/m/023t15	The Shooting Star	Hergé	1942		 One particularly hot evening Tintin is out walking with his dog Snowy. Tintin then notices an extra star in the Great Bear. When he reaches home, he calls the observatory. They say that they have the phenomenon under observation and hang up. From his window, Tintin sees that the star is getting bigger every minute. He walks to the observatory and, after some trouble, gets inside. He meets a man called Philippulus who proclaims himself to be a proph
           """

@pytest.fixture
def books_file(sample_books):
  with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(sample_books)
        return f.name

@pytest.fixture
def sample_books_chunks(sample_books):
  document = Document(page_content=sample_books)
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  chunks = splitter.split_documents([document]) 
  return chunks

