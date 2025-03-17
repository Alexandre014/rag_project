
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import time

app = FastAPI()

# Define OpenAI-compatible request model
class OpenAIRequest(BaseModel):
    model: str
    messages: list  # List of message history
    index_path: str = "indexes/global_index"
    docs_max: int = 6
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

@app.post("/v1/chat/completions")
def openai_chat(request: OpenAIRequest):
    
    #first request : autocompletion
    #'content': '### Task:\nYou are an autocompletion system. Continue the text in `<text>` based on the **completion type** in `<type>` and the given language.
    
    #second request : summarize the topic
    #'content': '### Task:\nGenerate a concise, 3-5 word title
    
    #third request : generate tags
    #'content': '### Task:\nGenerate 1-3 broad tags
    
    #if request contains'tag' return ...
    # if request contains'Generate a concise, 3-5 word title'
    
    print("REQUEST:", request)
    # Extract user question from messages
    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # Load index dynamically
    db = load_index(request.index_path)
    retriever = db.as_retriever(search_kwargs={"k": request.docs_max})

    # Retrieve relevant documents
    docs = retriever.invoke(user_message)
    if not docs:
        return {
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"message": {"role": "assistant", "content": "Je ne parviens pas à répondre à partir de ces documents."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }

    # Format context
    context = " ".join([doc.page_content for doc in docs])

    # Define the prompt
    prompt = (
        "You are a French AI assistant answering questions based strictly on the provided documents. "
        "Your response should be in French, concise, accurate, and directly relevant to the question. "
        "If the documents do not contain enough information, say 'Je ne parviens pas à répondre à partir de ces documents.' "
        "\n\n"
        "### Documents:\n"
        "{context}"
        "\n\n"
        "### Question:\n"
        "{question}"
    )

    # Create conversation messages
    messages = [{'role': 'user', 'content': prompt.format(context=context, question=user_message)}]
    print("MessagesPrompt", messages)
    # Query the LLM
    response = query_ollama(request.model, messages, request.ollama_server_url)
    
    print(response['message']['content'])
    # Format response in OpenAI format
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"message": {"role": "assistant", "content": response['message']['content']}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }

    
@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-r1:8b", "object": "model"},
            {"id": "deepseek-r1:32b", "object": "model"},
            {"id": "llama3.2", "object": "model"},
            {"id": "mistral", "object": "model"},   
        ]
    }