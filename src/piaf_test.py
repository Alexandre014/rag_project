from datasets import load_dataset
from datasets import get_dataset_split_names

from langchain_community.document_loaders import HuggingFaceDatasetLoader
#piaf_dataset = load_dataset("AgentPublic/piaf")
#piaf_dataset.save_to_disk("./local_piaf_dataset")
#print(piaf_dataset["train"][0])

loader = HuggingFaceDatasetLoader("AgentPublic/piaf", "context")
data = loader.load() # Load the data
print(data[0].page_content[10:80])
#here i want to reduce data amount
#data = data[:1000]
data = list({doc.page_content[:80]: doc for doc in data}.values())
print(len(data))
