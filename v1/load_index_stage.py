import sys
import asyncio
import tkinter as tk
from tkinter import filedialog
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

file_path = ".\data\pdf\consignes_stage.pdf"

loader = PyPDFLoader(file_path)

# Load the data
async def load_pages(loader):
    pages = []
    async for page in loader.alazy_load(): 
        pages.append(page)
    return pages

data = asyncio.run(load_pages(loader))

#print(f"{data[0].metadata}\n")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)



# model for retrieving
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


#vector store
db = FAISS.from_documents(docs, embeddings)
db.save_local("indexes")
