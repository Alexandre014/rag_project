import os
import requests
import json
import sys
import load_index
import tkinter as tk
from tkinter import filedialog
import asyncio
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from ollama import chat
from ollama import ChatResponse

#OLLAMA_SERVER_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_SERVER_URL = "https://tigre.loria.fr:11434/api/chat"

docs_max = 4 #maximum number of documents retrieved and used to generate an answer (documents not files)

"""send a query to Ollama server"""
def query_ollama(model, messages):
    payload = {
        "model": model,
        "messages": [{"role": message['role'], "content": str(message['content'])} for message in messages],
        "stream": False 
    }
    print(messages)
    try:
        response = requests.post(OLLAMA_SERVER_URL, json=payload)
        #response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error to ollama : {e}")
        return "Error while generating the response."


"""launch the rag on a specified index"""
def launch_rag(index_location, generation_model="llama3.2" ):
    #step 1 : load embedding

    modelPath = "sentence-transformers/all-MiniLM-l6-v2" # model for embedding

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
    db = FAISS.load_local(index_location, embeddings=embeddings, allow_dangerous_deserialization= True)



    #step 2 : retrieve data and start conversation

    retriever = db.as_retriever(search_kwargs={"k": docs_max})

    question = input("Posez votre question : ")

    prompt = (
        "You are a french AI assistant answering questions based strictly on the provided documents. "
        "Your response should be in french, concise, accurate, and directly relevant to the question. "
        "If the documents do not contain enough information, say 'Je ne parviens pas à répondre à partir de ces documents.' "
        "\n\n"
        "### Documents:\n"
        "{context}"
        "\n\n"
        "### Question:\n"
        "{question}"
        )

    conversation=[]

    while question != "quit":

        docs = retriever.invoke(question)
        print("SIZE : ",len(docs), "FIRST DOC: ", docs[0])
        context = " ".join([doc.page_content for doc in docs])  # concatenate text

        conversation.append({'role': 'user', 'content': prompt.format(context=context, question=question)})
   
        response = query_ollama(generation_model, conversation)

        print("\n", question, "\n")
        print(response['message']['content'])
        conversation.append({'role': 'assistant', 'content': response})
        
        
        print("\nLa réponse a été générée à partir des ", len(docs), " documents suivants : ")
        for doc in docs:
            print(os.path.basename(doc.metadata['source']), ": page", doc.metadata['page_label'])
        question = input("Posez votre question : ")
    



def main():
    index_path = "indexes/global_index"
    #load_index.load_index_from_directory("data/pdf", index_path) #load the index only one time if you don't change the files
    launch_rag(index_path, "llama3.2")

if __name__ == "__main__":
    main()
    