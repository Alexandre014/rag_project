import rag


def main():
    index_path = "indexes/dolly_index" 
    rag.launch_rag(index_path, "mistral")

if __name__ == "__main__":
    main()