import sys
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

file_path = ".\data\pdf\consignes_stage.pdf"

# Create a loader instance
loader = PyPDFLoader(file_path)


# Load the data
async def load_pages(loader):
    pages = []
    async for page in loader.alazy_load(): 
        pages.append(page)
    return pages

data = asyncio.run(load_pages(loader))

#print(f"{data[0].metadata}\n")

# Display the two first entries
#print("two first entries: ", data[:2])


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)

#print("first chunk: ", docs[0])


# Define the path to the pre-trained model you want to use
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



question = "Résume moi ce document"
#question = input("Pose ta question : ")

#vector store
db = FAISS.from_documents(docs, embeddings)
#use save_local() and load_local() to save the vector as an index





#step 2

#searchDocs = db.similarity_search(question) #for search solely on embedings

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

docs = retriever.get_relevant_documents(question)
context = " ".join([doc.page_content for doc in docs])  # concatenate text


response: ChatResponse = chat(
    model='llama3.2',
    messages=[
        {'role': 'assistant', 'content': context},#give the context
        {'role': 'user', 'content': question},
    ]
)



print("\n", question, "\n")
print(response.message.content)