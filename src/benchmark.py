import requests
import csv
import time
import os

API_URL = "http://127.0.0.1:8000/v1/chat/completions" #the RAG API url
QUESTIONS_FILE = "eval_questions.txt"
MODEL_NAME = "deepseek-r1:32b" # model used for the tests
CSV_OUTPUT = "./benchmarks/" + MODEL_NAME.replace(":", "_") + "_benchmark_results.csv" # responses file name
INDEX_PATH = "indexes/pdf_indexes/e5base" # index storing the documents

# if the file name is already used
if os.path.exists(CSV_OUTPUT):
    erase = input("This benchmark already exists, do you want to delete it? (y/n)")
    if erase != "y" : 
        raise Exception("This benchmark already exists, benchmark canceled")
    
# Retrieve questions
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]

# Generate benchmark
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Question", "Response", "Response time (s)"])

    for question in questions:
        print(f"Sending question : {question}")

        # Model configuration
        request_data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": question}],
            "index_path": INDEX_PATH,
            "docs_max": 6,
            "ollama_server_url": "https://tigre.loria.fr:11434/api/chat"
        }

        start_time = time.time()

        try:
            # send the request
            response = requests.post(API_URL, json=request_data)
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"Error: {e}"

        elapsed_time = round(time.time() - start_time, 2)

        if("Error: HTTPConnectionPool" in answer):
            raise Exception("API error")
        
        print(answer)
        
        #write to the csv file
        writer.writerow([question, answer, elapsed_time])
        print(f"Response received in {elapsed_time}s\n")

print(f"Benchmark completed.")
