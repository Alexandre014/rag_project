import os
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import HuggingFaceDatasetLoader

def get_all_files_in_directory(directory_path, extension='.pdf'):
    """Get the path of all files in a directory"""
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Load the data
async def load_pages(loader):
    pages = []
    async for page in loader.alazy_load(): 
        pages.append(page)
    return pages


def embed_data(data, index_destination):
    """Embed the data and create an index"""
      # Data splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    # Define the retrieving configuration for later
    modelPath = "intfloat/multilingual-e5-base" # model for retrieving
    model_kwargs = {'device':'cpu'} # model configuration options
    encode_kwargs = {'normalize_embeddings': False} # encoding options

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    #vector store
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_destination)

def load_index_from_directory(directory_path, index_destination):
    """Load files from a directory as an index and store it at a specified location"""
    all_pages = []

    # Get the paths
    file_paths_list = get_all_files_in_directory(directory_path)

    # Load documents
    for file_path in file_paths_list:
            loader = PyPDFLoader(file_path)
            try:
                data = asyncio.run(load_pages(loader)) 
                all_pages.extend(data) 
                print(f"Successfully loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    embed_data(all_pages, index_destination)

def load_index_from_dataset(dataset_name, data_column, index_destination):
    """Load a dataset as an index and store it at a specified location"""
    
    loader = HuggingFaceDatasetLoader(dataset_name, data_column)
    data = loader.load() # Load the data
    
    #print(data[0].page_content)
    #here i want to reduce data amount
    #data = data[:1000]
    
    # We keep each value only once
    #unique_data = list({doc.page_content: doc for doc in data}.values())
    unique_dict = {}
    for doc in data:
        unique_dict[doc.page_content] = doc
        unique_dict[doc.page_content].page_content += " title : " + unique_dict[doc.page_content].metadata.get('title', 'No Title') +";"
        
    unique_data = list(unique_dict.values()) 
    
    #unique_data = data[:100]
    print(data[0])
    print(len(unique_data))
    embed_data(unique_data, index_destination) 
    
        

def main():
    
    index_path = "indexes/dataset_indexes/e5base_Full"
    #load_index_from_directory("data/pdf", index_path) #load the index only one time if you don't change the files
    load_index_from_dataset("AgentPublic/piaf", "context", index_path)

if __name__ == "__main__":
    main()