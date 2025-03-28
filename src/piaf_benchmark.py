import requests
import csv
import time
import os
from datasets import load_dataset
import openai_rag_api
API_URL = "http://127.0.0.1:8000/v1/chat/completions" #the RAG API url
MODEL_NAME = "mistral" # model used for the tests
CSV_OUTPUT = "./benchmarks/piaf_e5base_camembert_" + MODEL_NAME.replace(":", "_") + "_benchmark_results.csv" # responses file name
INDEX_PATH = "indexes/dataset_indexes/piaf_e5base_Full" # index storing the documents
QUESTIONS_AMOUNT = 15

DATASET_NAME = "AgentPublic/piaf"
EXPECTED_ANSWER_COLUMN = "answers"

# if the file name is already used
if os.path.exists(CSV_OUTPUT):
    erase = input("This benchmark already exists, do you want to delete it? (y/n)")
    if erase != "y" : 
        raise Exception("This benchmark already exists, benchmark canceled")
    
# to retrieve questions
dataset = load_dataset("AgentPublic/piaf", "plain_text", split="train")
dataset = dataset.select(range(QUESTIONS_AMOUNT))

# Generate benchmark
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Question", "Response", "Response time (s)", "Answer quality"])

    for question, expected_answer in zip(dataset['question'], dataset[EXPECTED_ANSWER_COLUMN]):
        print(f"Sending question : {question}")

        # retrieving model configuration
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
        
        print("Expected answer : ", expected_answer["text"][0])
        # evaluate the answer quality
        answer_quality = openai_rag_api.evaluate_answer_quality_camembert(answer, expected_answer["text"][0]) 
        print(answer_quality)
    
        
        #write to the csv file
        writer.writerow([question, answer,  answer_quality, expected_answer["text"][0], elapsed_time])
        print(f"Response received in {elapsed_time}s\n")

print(f"Benchmark completed.")
