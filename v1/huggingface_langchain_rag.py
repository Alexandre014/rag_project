import sys
import asyncio
import torch
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
db = FAISS.load_local("indexes", embeddings=embeddings, allow_dangerous_deserialization= True)

question = "Dans quel format faut il faire la lettre au début du stage ?"
#question = input("Pose ta question : ")


#step 2



# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
# retriever = db.as_retriever(search_kwargs={"k": 4})
# docs = retriever.invoke(question)
docs = db.similarity_search(question)
context = " ".join([doc.page_content for doc in docs])  # concatenate text

print("\nCONTEXT \n", context)



prompt = f"Answer the question by using this context : {context} \n\n Question : {question}"


print("\nPROMPT \n", prompt)
raw_inputs = [
    prompt,
]

"""tokenization"""
tokenizer = AutoTokenizer.from_pretrained("model", low_cpu_mem_usage=True)


inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt")

print("\INPUTS \n", inputs)


"""model"""
model = AutoModelForCausalLM.from_pretrained("model", low_cpu_mem_usage=True) #model for generation

outputs = model(**inputs)
print(outputs)# logits
