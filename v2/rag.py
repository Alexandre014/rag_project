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

docs_max = 4 #maximum number of documents retrieved and used to generate an answer (documents not files)

"""launch the rag on a specified index"""
def launch_rag(index_location):
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
    db = FAISS.load_local("indexes/global_index", embeddings=embeddings, allow_dangerous_deserialization= True)



    #step 2 : retrieve data and start conversation

    retriever = db.as_retriever(search_kwargs={"k": docs_max})

    question = input("Posez votre question : ")

    prompt = (
        "You are a french AI assistant answering questions based strictly on the provided documents. "
        "Your response should be concise, accurate, and directly relevant to the question. "
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
        
        response: ChatResponse = chat(
            model='llama3.2',
            messages=conversation
        )

        print("\n", question, "\n")
        print(response.message.content)
        
        conversation.append({'role': 'assistant', 'content': response.message.content})
        
        question = input("Posez votre question : ")
    



def main():
    index_path = "indexes/global_index"
    #load_index.load_index_from_directory("data/pdf", index_path) #load the index only one time if you don't change the files
    launch_rag(index_path)

if __name__ == "__main__":
    main()
    