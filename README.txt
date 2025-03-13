RAG PROJECT

How to launch the server:
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload

Post request : 
#make the request body: 
$body = @{
    question = "Qu'est-ce que le bizutage?"
    model = "llama3.2"
    index_path = "indexes/global_index"
    docs_max = 3
} | ConvertTo-Json -Depth 10

#send the request :
 $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" -Method Post -Headers @{"Content-Type"="application/json"} -Body $body

#display the response : 
 $response | ConvertTo-Json -Depth 10

