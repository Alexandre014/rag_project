import requests
import json

# Define the API endpoint
OLLAMA_SERVER_URL = "http://localhost:11434/api/generate"

# Define the payload (the body of the request)
payload = {
    "model": "llama3.2",
    "prompt": "coucou"
}

# Send the POST request to the server
response = requests.post(OLLAMA_SERVER_URL, json=payload, headers={"Content-Type": "application/json"})

# Check the raw content first
print("Raw Response Text:")
print(response.text)

# Try to split and parse each JSON object
try:
    # Split the response by newline to separate the JSON objects
    response_parts = response.text.strip().split("\n")
    
    # Parse each part separately
    for idx, part in enumerate(response_parts):
        print(f"Parsing response {idx + 1}:")
        response_json = json.loads(part)
        print("Parsed JSON:", response_json)
except json.JSONDecodeError as e:
    print("Error parsing JSON:", e)
    print("Response Text:", response.text)
