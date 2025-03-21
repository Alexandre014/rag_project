from datasets import load_dataset
from datasets import get_dataset_split_names

piaf_dataset = load_dataset("AgentPublic/piaf")
piaf_dataset.save_to_disk("./local_piaf_dataset")
print(piaf_dataset["train"][0])


