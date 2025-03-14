# RAG PROJECT

## Windows Powershell :

### How to launch the api server:

`uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload`

### Launch openwebui:

`open-webui server`
use the port 8080 when installed with python

### How to use the OpenAI api :

#### Post request :

```
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/chat/completions" -Method Post -ContentType "application/json" -Body (@{
    model="deepseek-r1:8b";
    messages=@(@{role="user"; content="Qu'est-ce que le bizutage ?"});
    index_path="indexes/global_index";
    docs_max=5;
    ollama_server_url="https://tigre.loria.fr:11434/api/chat"
} | ConvertTo-Json -Depth 10)
```

#### Display the response :

`$response | ConvertTo-Json -Depth 10`

### How to use the Ollama api :

#### Post request :

##### make the request body:

```
$body = @{
    question = "Qu'est-ce que le bizutage?"
    model = "llama3.2"
    index_path = "indexes/global_index"
    docs_max = 3
} | ConvertTo-Json -Depth 10
```

#### send the request :

`$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" -Method Post -Headers @{"Content-Type"="application/json"} -Body $body`

#### display the response :

`$response | ConvertTo-Json -Depth 10`
