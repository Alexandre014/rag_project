import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import time
import re
from ragas import SingleTurnSample 
from ragas.metrics import ResponseRelevancy
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

app = FastAPI() # FastAPI instance 

class OpenAIRequest(BaseModel):
    """Define OpenAI-compatible request"""
    model: str
    messages: list  # List of message history
    index_path: str = "indexes/piaf_100_index"
    docs_max: int = 6 # maximum number of documents retrieved and used to generate an answer (documents not files)
    ollama_server_url: str = "https://tigre.loria.fr:11434/api/chat" # choose your Ollama server, or localhost: http://127.0.0.1:11434/api/chat

def load_index(index_path):
    """Function to dynamically load an index"""
    
    if not os.path.exists(index_path):
        raise HTTPException(status_code=400, detail=f"Index not found at {index_path}")

    # embedding
    model_path = "sentence-transformers/all-MiniLM-l6-v2" 
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    
    return FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True) # allow_dangerous_deserialization=True -> Warning : load trustworthy files

def query_ollama(model, messages, ollama_server_url):
    """Function to query Ollama server"""
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False # False : to get the response in one shot
    }
    try:
        response = requests.post(ollama_server_url, json=payload)
        return response.json()
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying Ollama: {e}")

def format_response(request : OpenAIRequest, response):
    """format the API response into the OpenAI structure"""
    
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"message": {"role": "assistant", "content": response['message']['content']}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }

def base_prompt():
    return(
        "You are a French AI assistant answering questions based strictly on the provided documents. "
        "Your response should be in French, concise, accurate, and directly relevant to the question. "
        "If the documents do not contain enough information, say 'Je ne parviens pas à répondre à partir de ces documents.' "
    )

def generate_prompt(context, user_message):
    prompt = (
        base_prompt() +
        "\n\n"
        "### Documents:\n"
        "{context}"
        "\n\n"
        "### Question:\n"
        "{question}"
    )
    return prompt.format(context=context, question=user_message)

async def evaluate_relevancy(scorer, sample):
    score = await scorer.single_turn_ascore(sample)
    print(f"The relevancy score is: {score}")

@app.post("/v1/chat/completions")
def openai_chat(request: OpenAIRequest):
    """Respond to OpenAI chat queries"""
    
    #if the query is not a user query (for specific Openwebui queries)
    if ("### Task:" in request.messages[-1]['content'][:10]):
        # we just let Ollama respond
        return format_response(request, query_ollama(request.model, request.messages, request.ollama_server_url))
    
    # Extract user question from messages, so the latest user message
    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # Load index dynamically
    db = load_index(request.index_path)
    retriever = db.as_retriever(search_kwargs={"k": request.docs_max})

    # Retrieve relevant documents
    docs = retriever.invoke(user_message)
    if not docs:
        raise HTTPException(status_code=400, detail="No file found")

    print(docs)
    
    # Concatenate documents
    context = " ".join([doc.page_content for doc in docs])

    # Create Ollama query
    ollama_query = [{'role': 'user', 'content': generate_prompt(context, user_message)}]
    
    # Query the LLM
    llm_response = query_ollama(request.model, ollama_query, request.ollama_server_url)

    print(f"Response : {llm_response['message']['content']}")
    
    # Remove the "thinking" part from deepseek responses
    if ("deepseek" in request.model):
        llm_response['message']['content'] = re.sub(r"<think>.*?</think>\n?", "", llm_response['message']['content'], flags=re.DOTALL)
        
        
    sample = SingleTurnSample(
        user_input=user_message,
        response=llm_response['message']['content'],
        retrieved_contexts=[
            context
        ]
    )

    # evaluator_model = "mistralai/Mistral-7B-Instruct-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained(evaluator_model)
    # evaluator_llm = AutoModelForCausalLM.from_pretrained(evaluator_model, low_cpu_mem_usage=True, local_files_only=True, device_map="auto",  trust_remote_code=True)
    # pipe = pipeline("text-generation", model=evaluator_model, tokenizer=tokenizer)
    # evaluator_llm = HuggingFacePipeline(pipeline=pipe)
    
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # evaluator_llm = Ollama(model="mistral", base_url=request.ollama_server_url)
    evaluator_llm = ChatOllama(model="llama3")
    embeddings = OllamaEmbeddings(model="llama3")
   
    scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings)
    asyncio.run(evaluate_relevancy(scorer, sample))
    
    # Format response in OpenAI format
    return format_response(request, llm_response)

    
@app.get("/v1/models")
def get_models():
    """Return the available models list"""
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-r1:8b", "object": "model"},
            {"id": "deepseek-r1:32b", "object": "model"},
            {"id": "llama3.2", "object": "model"},
            {"id": "mistral", "object": "model"},   
        ]
    }