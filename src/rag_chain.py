from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

from .vector_store import BooksVectorStore


class BooksRAGChain:
    def __init__(self, vector_store: BooksVectorStore):
        """Initialize RAG chain with vector store.
        TODO: Implement initialization
        """
        self.vector_store = vector_store
        self.chain = None
        self._create_chain()


    def _create_chain(self):
        """Create the RAG chain.
        TODO: Implement this method
        """
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=300,
        )

        # Define a simple prompt template
        prompt = ChatPromptTemplate.from_template(
            "You are an expert movie assistant. Based on the following context, answer the question accurately.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )

        # Combine the components into a chain
        self.chain = {
            "llm": llm,
            "prompt": prompt,
            "parser": StrOutputParser(),
        }

    async def query(self, question: str) -> str:
        """Query the RAG chain.
        TODO: Implement this method
        """
        relevant_docs = self.get_relevant_documents(question)
        print(relevant_docs[0].page_content)
        inputs = {
            "context": relevant_docs,
            "question": question
        }
        prompt_text = self.chain["prompt"].format(**inputs)
        response = await self.chain["llm"].apredict(prompt_text)
        return response

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents with metadata.
        TODO: Implement this method
        """
        relevant_docs = self.vector_store.similarity_search(query, k)
        return relevant_docs
