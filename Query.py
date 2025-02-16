import argparse
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from Embedding import get_embeddings
from langchain.prompts import ChatPromptTemplate
#from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    print(query_rag(query_text))

def query_rag(query:str)-> str:
    #fetch context from vector database related to given query and pass it to llm.
    embedding_function = get_embeddings()
    db = Chroma(persist_directory = "chromaDB/", embedding_function = embedding_function)
    results = db.similarity_search(query, k= 3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query)
    model = OllamaLLM(model='llama3.2')
    response = model.invoke(prompt)    
    return str(response)

if __name__ == "__main__":
    main()