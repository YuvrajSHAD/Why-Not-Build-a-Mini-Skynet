from langchain_ollama import OllamaLLM

def main():
    # Initialize the LLM with the desired model
    llm = OllamaLLM(model="llama3.1")

    try:
        # Invoke the model with a prompt
        response = llm.invoke("The first man on the moon was ...")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
