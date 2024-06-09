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



# creating a pdf reader object 
reader = PdfReader(os.getcwd()+"/pdfs/policy-booklet-0923.pdf") 
  
# printing number of pages in pdf file 
print(len(reader.pages)) 
  
# creating a page object 
# for i in range(2,len(reader.pages)):
#     page = reader.pages[i] 
  
#     text = page.extract_text()
#     print(text)
   
#     text = text.replace('\n',' ').split('?')
#     print((text))
#     text_embeddings = np.array([get_text_embedding(chunk) for chunk in text])


page = reader.pages[2] 
  
text = page.extract_text()
print(text)

text = text.replace('\n',' ').split('?')
print((text))
client = chromadb.PersistentClient(os.getcwd())
collection = client.create_collection(name="docs")
# store each document in a vector embedding database
for i, d in enumerate(text):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )   
   # questions= []
#     answers = []
#     answer=''
#     for line in text:
#         if line[-1] == '?':
#             questions.append(line)
#             print(line)
#             print(answer)
#             answers.append(answer)
#             answer = ''
#         else:
#             answer = answer + line
        
# print(len(questions))
# print(len(answers))