# RAG Project with Langchain integration

## 📝 Overview

This project is a **Retrieval-Augmented Generation (RAG) system** using Langchain for embeddings and FAISS indexing. It allows querying an LLM with context retrieved from indexed documents.

## ✨ Features

-   📂 Loads and indexes PDF documents into a FAISS vector store.
-   🔎 Uses `intfloat/multilingual-e5-base` for text embeddings.
-   🚀 Provides a **FastAPI-based chat API** compatible with OpenAI-style requests.
-   📜 Retrieves and formats context from indexed documents.
-   🤖 Queries an **Ollama** server for responses.

## ⬇️ Installation

### 🔧 Prerequisites

-   A running **[Ollama](https://github.com/ollama/ollama?tab=readme-ov-file)** server (remote or local) :
    `ollama serve`

### 📦 Install Dependencies

Run the following command to install required dependencies:  
`pip install -r requirements.txt`

## 🚀 Usage

### 💬 Interaction Methods

You can interact with the system in multiple ways:

-   🖥️ Terminal: Run rag.py for direct testing.
-   🔗 API: Use openai-rag_api.py to query via an API.
-   🌐 Web Interface: Integrate with OpenWebUI for a user-friendly interface.

### 1️⃣ Index Documents (To be done only once, when you don't want to change the documents)

Add your PDF files to "data/pdf" folder.

Run the `load_index.py` script :

`python index_loader/load_index.py`

This will generate a FAISS index and save it in indexes/global_index.

To do it only once, when you don't want to change the documents.

### 2️⃣ Start API Server

Run the FastAPI server:
`uvicorn src.openai_rag_api:app --host 0.0.0.0 --port 8000`

### 3️⃣ Query API

### 🔹 OpenAI-Compatible Chat Endpoint

#### Windows Powershell :

Post request :

```
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/chat/completions" -Method Post -ContentType "application/json" -Body (@{
    model="Model name";
    messages=@(@{role="user"; content="Q"});
    index_path="indexes/global_index";
    docs_max=6;
    ollama_server_url="server url"
} | ConvertTo-Json)
```

Display the response :

`$response | ConvertTo-Json`

#### Linux :

Post request :

```
response=$(curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
          "model": "Model name",
          "messages": [{"role": "user", "content": "Q"}],
          "index_path": "indexes/global_index",
          "docs_max": 6,
          "ollama_server_url": "server url"
        }')
```

Display the response :
`echo "$response"`

### 🔹 Available Models Endpoint

#### Windows Powershell :

`Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/models" -Method Get`

#### Linux :

`curl -X GET "http://127.0.0.1:8000/v1/models"`

## 🤖 Openwebui integration

### 🔧 Prerequisites

-   A running **[Open Webui](https://docs.openwebui.com)** server :
    `open-webui serve`

### Configuration steps :

-   1️⃣ Go to Settings -> Admin Settings -> Connections

-   2️⃣ Desactivate "Ollama API"

-   3️⃣ Activate "Direct Connections"

-   4️⃣ In "Manage API OpenAI connections" add a new connection :

    -   In the URL field : http://127.0.0.1:8000/v1

    -   For the key and prefix, choose what you want.

### Choose your model and use it !

![Open WebUI user interface](image-1.png)
