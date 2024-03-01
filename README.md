# Install


## Python (install)
- Create the project directory
- Go to the project directory
- python3 -m venv venv
- Check if directory venv has been created
- pip install ollama
- To extra feature:
	 pip install langchain beautifulsoup4 chromadb gradio

### How to activate
- source venv/bin/activate

## Llama (install)
- Go to ollama webpage and download the server
- Add +x permission to the binary downloaded
- Execute as root the binary `ollama serve`
- Execute as regular user the binary ollama run <model>
	- Example of models:
		- llama2
		- mistral
- Wait its pull

### How to execute
- Execute as root the binary `ollama serve`
- After, execute the target python script to use ollama'
