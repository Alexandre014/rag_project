import load_index

def main():
    piaf_index_path="indexes/piaf_index"
    load_index.load_index_from_dataset("AgentPublic/piaf", "context", piaf_index_path)
    

if __name__ == "__main__":
    main()