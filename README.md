# Book Search and Q&A System Using RAG
## Problem Description

Create a Book Search and Question-Answering system using Retrieval-Augmented Generation (RAG) that allows users to search for books and ask questions about them. The system should use a combination of vector search and language models to provide accurate and contextual responses.

### Requirements

1. **Data Processing**
   - Use the provided books dataset sample (1000 books)
   - Process Book summary information 
   - Create embeddings for efficient search

2. **Core Features**
   - Search for books by description
   - Answer questions about specific books
   - Handle natural language queries effectively

3. **Technical Requirements**
   - Use LangChain for RAG implementation
   - Use FAISS for vector storage
   - Use OpenAI's GPT-4 (use "gpt-4o-mini") for text generation

4. **Implementation Details**
   - Implement proper error handling
   - Ensure efficient vector search

### Tasks to Complete

1. Implement the `BooksLoader` class in `loader.py`
2. Implement the `BooksVectorStore` class in `vector_store.py`
3. Implement the `BooksRAGChain` class in `rag_chain.py`
4. Make sure tests passes for all features.

**NOTE:** Implement proper error handling FileNotFoundError as this scenario is covered in the unit tests.

### Sample Data Format
```text
843	/m/0k36	A Clockwork Orange	Anthony Burgess	1962	{"/m/06n90": "Science Fiction", "/m/0l67h": "Novella", "/m/014dfn": "Speculative fiction", "/m/0c082": "Utopian and dystopian fiction", "/m/06nbt": "Satire", "/m/02xlf": "Fiction"}	 Alex, a teenager living in near-future England, leads his gang on nightly orgies of opportunistic, random "ultra-violence." Alex's friends ("droogs" in the novel's Anglo-Russian slang, Nadsat) are: Dim, a slow-witted bruiser who is the gang's muscle

986	/m/0ldx	The Plague	Albert Camus	1947	{"/m/02m4t": "Existentialism", "/m/02xlf": "Fiction", "/m/0pym5": "Absurdist fiction", "/m/05hgj": "Novel"}	 The text of The Plague is divided into five parts. In the town of Oran, thousands of rats, initially unnoticed by the populace, begin to die in the streets. A hysteria develops soon afterward, causing the local newspapers to report the incident. Authorities responding to public pressure order the collection and cremation of the rats,
```
