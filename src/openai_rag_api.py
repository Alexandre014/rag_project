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

from langdetect import detect
from deep_translator import GoogleTranslator

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
    return (
        "You are a French AI assistant answering questions based strictly on the provided documents. "
        "Your response should be in French, concise, accurate, and directly relevant to the question. "
        "If the documents do not contain enough information, say 'Je ne parviens pas à répondre à partir de ces documents.' "
    )
def dataset_prompt():
    return (
        "You are a French AI assistant tasked with answering questions strictly based on the provided documents. "
        "Your response must be an exact excerpt from the document—a word or a short phrase—without any modification or additional generation. "
        "The answer should be as concise as possible."
        "\n\n"
        "Examples:\n"
        "- Question: Quelle est la position de la marine indienne en termes d'effectifs à l'échelle mondiale ?\n"
        "  Response: quatrième\n"
        "- Question: Combien la marine indienne a-t-elle de porte-avions en service ?\n"
        "  Response: un\n"
        "- Question: Quel est l'effet de l'acide fusidique ?\n"
        "  Response: bloquent par exemple la translocation\n"
        "- Question: Quels organismes sont surtout concernés par le blocage de la traduction ?\n"
        "  Response: les bactéries\n\n"
        "Do not include the document title in your response."
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
    stop_words = stopwords.words('french')

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

    return similarity_score.item()

def clean_original_response(text):
    """Keep the first response before any rephrasing"""
    patterns = [
        r"(?i)^\**\s*réponse(\s+en\s+français)?(\s+(concise|finale|formatée))?\s*[:：]",
    ]

    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if any(re.match(pattern, line.strip()) for pattern in patterns):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def verify_language(text):
    """return false if the text contains wrong characters or if the first sentence is in english"""
    # \u4e00-\u9fff → chinese characters
    # \u0400-\u04FF → cyrillic characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    cyrillic_pattern = re.compile(r'[\u0400-\u04FF]')

    if chinese_pattern.search(text) or cyrillic_pattern.search(text):
        return False
    
    #sentences = re.split(r'[.?!]\s+', text)
    #first_sentence = sentences[0] if sentences else text

    try:
        lang = detect(text)
        if lang == 'en':
            return False
    except:
        print("LANGUAGE ERROR")
        return False

    return True

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
    context = " ".join([doc.page_content + "document title:" + doc.metadata["source"]+ "; next document:" for doc in docs])

    # Create Ollama query
    evaluation = False
    if ("dataset" in request.index_path):
        evaluation = True
    ollama_query = [{'role': 'user', 'content': generate_prompt(context, user_message, evaluation)}]
    
    correct_language = False
    
    total_attempts = 0 
    while not correct_language and total_attempts < 3:
        total_attempts+=1 # after 3 atempts we keep the last wrong response
        
        # Query the LLM
        llm_response = query_ollama(request.model, ollama_query, request.ollama_server_url)
        
        print(f"Response : {llm_response['message']['content']}")
        
        # Remove the "thinking" part from deepseek responses
        if ("deepseek" in request.model):
            llm_response['message']['content'] = re.sub(r"<think>.*?</think>\n?", "", llm_response['message']['content'], flags=re.DOTALL)
            
        if verify_language(llm_response['message']['content']):
            correct_language = True
        
        # Remove a possible rephrasing
        llm_response['message']['content'] = clean_original_response(llm_response['message']['content'])
        
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