import os
import requests
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Serveur Ollama
OLLAMA_SERVER_URL = "https://tigre.loria.fr:11434"
#OLLAMA_SERVER_URL = "http://localhost:11434"

# Nombre de documents à récupérer
docs_max = 6  

def launch_rag(index_location, generation_model="mistral"):
    """Lance le RAG sur un index donné."""

    # Étape 1 : Charger les embeddings et l'index FAISS
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False}
    )
    db = FAISS.load_local(index_location, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": docs_max})

    # Étape 2 : Définir le modèle et le prompt
    llm = ChatOllama(base_url=OLLAMA_SERVER_URL, model=generation_model)

    prompt_template = PromptTemplate(
        template=(
            "You are a french AI assistant answering questions based strictly on the provided documents. "
            "Your response should be in french, concise, accurate, and directly relevant to the question. "
            "If the documents do not contain enough information, say 'Je ne parviens pas à répondre à partir de ces documents.' "
            "\n\n"
            "### Documents:\n"
            "{context}"
            "\n\n"
            "### Question:\n"
            "{question}"
        ),
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt_template}
    )

    # Étape 3 : Démarrer l'interaction avec l'utilisateur
    question = input("Posez votre question : ")
    while question != "quit":
        response = qa_chain.invoke(question)
        print("\nResponse:", response["result"], "\n")
        question = input("Posez votre question : ")

def main():
    index_path = "indexes/global_index"
    launch_rag(index_path, "llama3.2")

if __name__ == "__main__":
    main()
