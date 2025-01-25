### RAG (Retrieval-Augmented Generation) using LLM MODEL Setup

A Retrieval-Augmented Generation (RAG) chatbot created using LangChain and Hugging Face. This implementation is designed to work with any LLM (Large Language Model) and currently uses [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

This can be used for any general RAG use case

## Features

- Embedding Extraction done using [sentence-transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from hugging face and storing it in vector db.

- Knowledge Base uses FAISS vector database storage to perform semantic search and uses the context in the database to provide to rank result and the LLM provides the response to the user's prompt

- Any Knowledge can be provided in the context dir (books, pdfs, articles ) [in pdf format, can modify the context_maker.py file to accept more types] and the LLM uses semantic search to accurately answer based on the context

- Uses streamlit to server the RAG in order for quick test run and chat interface

## Setup and installation

Make sure you have

- Python3 installed.
- `pip` installed for package management.

1. Creation of virtualenv

```bash
python3 -m venv env
```

2. activate virtual env

```bash
env/Scripts/activate #Windows
source env/bin/activate #Liuux
```

3. install requirements.

```bash
pip install -r requirements.txt
```

4. Add your Knowledge Base

   4.1 Based on your requirements of chatbot add resources (pdf files) in the context folder

   4.2 PDFs of books, research papers, articles, documents,

   Optional -> in `context_maker.py` modify the `chunk_size` and `chunk_overlap` based on the size of your data to fine tune the embeddings based on your requirements.

   4.3 run `context_maker.py` file

5. Preapre to Load LLM and load the llm with memory of the vector database

   5.1 Make sure to use your huggingface API token after creating an account on [Hugging Face](https://huggingface.co/) and generating an API key with appropiate
   User permissions [enable repo and inference permissions]
   and then use that API key in code

   5.2 You can modify the parameters based on your requirements in `memory_with_llm.py`

6. Run the main.py file using streamlit to serve the RAG

```bash
streamlit run main.py
```
