from pypdf import PdfReader 
from groq import Groq
import json
import os
from io import BytesIO
import re

import ollama
import chromadb
import numpy as np
import os
import pathlib
import streamlit as st
import datetime

# creating a pdf reader object 
reader = PdfReader(os.getcwd()+"/pdfs/policy-booklet-0923.pdf") 
  
# printing number of pages in pdf file 
print(len(reader.pages)) 
client = chromadb.PersistentClient(os.getcwd())
# collection = client.create_collection(name="docs")
collection = client.get_collection(name='docs')
# creating a page object 
# if not os.path.exists(os.getcwd()+'\chroma.sqlite3'):
#     for j in range(2,len(reader.pages)):
#         page = reader.pages[j] 
#         text = page.extract_text()
#         print(text)
#         text = text.replace('\n',' ').split('?')
#         for i, d in enumerate(text):
#             response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
#             embedding = response["embedding"]
#             collection.add(
#                 ids=[str(i)],
#                 embeddings=[embedding],
#                 documents=[d]
#         )


prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")

