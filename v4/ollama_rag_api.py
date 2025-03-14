from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    question: str
    model: str = "deepseek-r1:8b"
    index_path: str = "indexes/global_index"
    docs_max: int = 4  # Default number of retrieved documents
    ollama_server_url: str = "https://tigre.loria.fr:11434/api/chat"

# Function to dynamically load an index
def load_index(index_path):
    if not os.path.exists(index_path):
        raise HTTPException(status_code=400, detail=f"Index not found at {index_path}")

    model_path = "sentence-transformers/all-MiniLM-l6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    
    return FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)

# Function to query Ollama
def query_ollama(model, messages, ollama_server_url):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False 
    }
    try:
        response = requests.post(ollama_server_url, json=payload)
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying Ollama: {e}")

@app.post("/query")
def ask_rag(request: QueryRequest):
    # Load index dynamically
    db = load_index(request.index_path)
    retriever = db.as_retriever(search_kwargs={"k": request.docs_max})

    # Retrieve relevant documents
    docs = retriever.invoke(request.question)
    if not docs:
        return {"question": request.question, "response": "Je ne parviens pas à répondre à partir de ces documents."}

    # Format context
    context = " ".join([doc.page_content for doc in docs])

    # Define the prompt
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

    # Create conversation messages
    messages = [{'role': 'user', 'content': prompt.format(context=context, question=request.question)}]
    
    # Query the LLM
    response = query_ollama(request.model, messages, request.ollama_server_url)
    
    return {
        "question": request.question,
        "model": request.model,
        "index_path": request.index_path,
        "response": response
    }
