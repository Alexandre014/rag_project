import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# choose the server you want to use 
#OLLAMA_SERVER_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_SERVER_URL = "https://tigre.loria.fr:11434/api/chat"

docs_max = 6 #maximum number of documents retrieved and used to generate an answer (documents not files)

def query_ollama(model, messages):
    """Send a query to Ollama server"""
    
    payload = {
        "model": model,
        "messages": [{"role": message['role'], "content": str(message['content'])} for message in messages],
        "stream": False 
    }
    
    try:
        response = requests.post(OLLAMA_SERVER_URL, json=payload)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error to ollama : {e}")
        return "Error while generating the response."


def launch_rag(index_location, generation_model="llama3.2" ):
    """Launch the rag on a specified index"""

    """step 1 : load embedding"""

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


    """step 2 : retrieve data and start conversation"""

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
        
        docs = retriever.invoke(question) # retrieved documents 
        context = " ".join([doc.page_content for doc in docs])  # concatenate documents
        
        print(context)
        
        conversation.append({'role': 'user', 'content': prompt.format(context=context, question=question)})
   
        response = query_ollama(generation_model, conversation)
        print("\nResponse: ", response['message']['content'], "\n")
        
        conversation.append({'role': 'assistant', 'content': response})
        
        # display source documents used to answer (file name + page number)
        if docs and ('source' in docs[0].metadata):
            print("\nLa réponse a été générée à partir des ", len(docs), " documents suivants : ")
            for doc in docs:
                print(os.path.basename(doc.metadata['source']), ": page", doc.metadata['page_label'])
            
        print(f"\nUse 'quit' to stop\n")
        question = input("Posez votre question : ")
    



def main():
    index_path = "indexes/global_index" 
    launch_rag(index_path, "mistral")

if __name__ == "__main__":
    main()
    