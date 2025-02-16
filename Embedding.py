#from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_embeddings():
    embedding = OllamaEmbeddings(model= "nomic-embed-text")
    return embedding