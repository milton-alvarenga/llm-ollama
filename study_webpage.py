import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

debug = True

if debug:
    print("Starting...")

loader = WebBaseLoader(
    web_paths=("https://devblog.drall.com.br/sample-page",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("entry-title","entry-content","entry-meta")
            #class_=("entry-title")
        )
    ),
)

if debug:
    print("Loading web_base_loader")

docs = loader.load()

if debug:
    print("Loading")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

if debug:
    print("Recursive Text Splitter")

splits = text_splitter.split_documents(docs)

if debug:
    print("Splits")

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="mistral")

if debug:
    print("Created embeddings")

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

if debug:
    print("Create vectorstore")

# Create the retriever
retriever = vectorstore.as_retriever()

if debug:
    print("Create retriever")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

if debug:
    print("Prepare to use RAG chain")
# Use the RAG chain
result = rag_chain("Who create the content of the blog?")
print(result)
result = rag_chain("What is the url of its news blog?")
print(result)
result = rag_chain("What are their site url?")
print(result)
result = rag_chain("Please resume in less than 24 words the article")
