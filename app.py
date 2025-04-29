import asyncio
import streamlit as st
from src.loader import BooksLoader
from src.vector_store import BooksVectorStore
from src.rag_chain import BooksRAGChain


def main():
    st.title("Personalized Book Discovery and Knowledge Assistant")
    st.write("Ask questions about books or search related information")

    query = st.text_input(
        "Enter the question or search query:",
        placeholder = "e.g., 'what are some famous writings of 1970s'"
    )

    loader = BooksLoader("data/books.txt")
    documents = loader.document_load()
    text_chunks = loader.create_chunks(documents)
    books_vector_store = BooksVectorStore()
    books_vector_store.create_vector_store(text_chunks)

    if st.button("Search") and query:
        with st.spinner("Searching for relevant books...."):
            rag_chain = BooksRAGChain(books_vector_store)
            response = asyncio.run(rag_chain.query(query))

            st.subheader("AI response")
            st.write(response)

if __name__ == "__main__":
    main()
