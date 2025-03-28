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

# evalution imports
import logging
from nltk.corpus import stopwords
from nltk import download
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import gensim.downloader as api
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex

from sentence_transformers import SentenceTransformer, util

app = FastAPI() # FastAPI instance 

class OpenAIRequest(BaseModel):
    """Define OpenAI-compatible request"""
    model: str
    messages: list  # List of message history
    index_path: str = "indexes/pdf_indexes/e5base"
    docs_max: int = 6 # maximum number of documents retrieved and used to generate an answer (documents not files)
    ollama_server_url: str = "https://tigre.loria.fr:11434/api/chat" # choose your Ollama server, or localhost: http://127.0.0.1:11434/api/chat

def load_index(index_path):
    """Function to dynamically load an index"""
    
    if not os.path.exists(index_path):
        raise HTTPException(status_code=400, detail=f"Index not found at {index_path}")

    # embedding
    model_path = "intfloat/multilingual-e5-base" 
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

def rag_prompt():
    return(
        "You are a French AI assistant answering questions based strictly on the provided documents. "
        "Your response should be in French, concise, accurate, and directly relevant to the question. "
        "If the documents do not contain enough information, say 'Je ne parviens pas à répondre à partir de ces documents.' "
    )
def dataset_prompt():
    return(
        "You are a french AI assistant answering questions based strictly on the provided documents. "
        "Your response should be a small chunk (one word or a string of words) of the document, answering directly to the question. Do not generate something by yourself"
        "The answer should be as short as possible."
        "For example : (Question: Quelle est la position de la marine indienne en termes d'effectifs à l'échelle mondiale ? Response: quatrième), (Question: Combien la marine indienne a-t-elle de porte-avions en service ? Response: un), (Question: Quel est l'effet de l'acide fusidique ? Response: bloquent par exemple la translocation), (Question: Quels organismes sont surtout concernés par le blocage de la traduction ? Response: les bactéries)"
        "Do not give the document title"
    ) 
    

def generate_prompt(context, user_message, evaluation_prompt=False):
    default_prompt = rag_prompt()
    if evaluation_prompt: default_prompt = dataset_prompt()
    prompt = (
        default_prompt +
        "\n\n"
        "### Documents:\n"
        "{context}"
        "\n\n"
        "### Question:\n"
        "{question}"
    )
    return prompt.format(context=context, question=user_message)

def evaluate_answer_quality_gensim(generated_answer, expected_answer):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # Import and download stopwords from NLTK.
    download('stopwords')  # Download stopwords list.
    stop_words = stopwords.words('english')

    def preprocess(sentence):
        return [w for w in sentence.lower().split() if w not in stop_words]

    generated_answer = preprocess(generated_answer)
    expected_answer = preprocess(expected_answer)

    documents = [generated_answer, expected_answer]
    dictionary = Dictionary(documents)

    generated_answer = dictionary.doc2bow(generated_answer)
    expected_answer = dictionary.doc2bow(expected_answer)

    documents = [generated_answer, expected_answer]
    tfidf = TfidfModel(documents)

    # generated_answer = tfidf[generated_answer]
    # expected_answer = tfidf[expected_answer]

    model = api.load('word2vec-google-news-300')

    termsim_index = WordEmbeddingSimilarityIndex(model)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)

    similarity = termsim_matrix.inner_product(generated_answer, expected_answer, normalized=(True, True))
    return 'similarity = %.4f' % similarity

def evaluate_answer_quality_camembert(generated_answer, expected_answer):
    model = SentenceTransformer("dangvantuan/sentence-camembert-base")

    embedding1 = model.encode(generated_answer, convert_to_tensor=True)
    embedding2 = model.encode(expected_answer, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(embedding1, embedding2)

    return 'similarity = %.4f' % similarity_score.item()


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
    evaluation = False
    if ("dataset" in request.index_path):
        evaluation = True
    ollama_query = [{'role': 'user', 'content': generate_prompt(context, user_message, evaluation)}]
    
    # Query the LLM
    llm_response = query_ollama(request.model, ollama_query, request.ollama_server_url)

    print(f"Response : {llm_response['message']['content']}")
    
    # Remove the "thinking" part from deepseek responses
    if ("deepseek" in request.model):
        llm_response['message']['content'] = re.sub(r"<think>.*?</think>\n?", "", llm_response['message']['content'], flags=re.DOTALL)
        
    
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