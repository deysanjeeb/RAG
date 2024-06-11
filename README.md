# RAG

## Description

This project uses the RAG (Retrieval-Augmented Generation) model for question answering tasks. It extracts text from PDF files, processes the text to extract question-answer pairs, and stores these pairs in a ChromaDB collection for later retrieval.

## Installation

This project requires Python and several Python libraries, including `chromadb`, `json`, and `re`. You can install these libraries using pip:

```sh
pip install -r requirements.txt
```

## Usage
To use this project, run the main.py script. This script will process all PDF files in the current directory and store the extracted question-answer pairs in a ChromaDB collection.

```sh
streamlit run main.py
```
