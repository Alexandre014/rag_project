import sys
import os
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

def get_all_files_in_directory(directory_path, extension='.pdf'):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Load the data
async def load_pages(loader):
    pages = []
    async for page in loader.alazy_load(): 
        pages.append(page)
    return pages


"""load files from a directory as an index and store it at a specified location"""
def load_index_from_directory(directory_path, index_destination):

    all_pages = []

    file_paths_list = get_all_files_in_directory(directory_path)


    for file_path in file_paths_list:
            loader = PyPDFLoader(file_path)
            try:
                data = asyncio.run(load_pages(loader)) 
                all_pages.extend(data) 
                print(f"Successfully loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    docs = text_splitter.split_documents(all_pages)

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
    db.save_local(index_destination)


    return index_destination


def main():
    index_path = "indexes/global_index"
    load_index_from_directory("data/pdf", index_path) #load the index only one time if you don't change the files
    

if __name__ == "__main__":
    main()