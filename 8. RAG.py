# RAG (Retrieval-Augmented Geneation)
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

# Embedding is the process of converting text(word, sentences, document) into a numerical vercot
# Then we can store them in vec DB or pwoer other function up

# embed the data into vec db -> retriever find the right chunks -> combine the ans with prompt -> retriever chain will deliever the answer out
load_dotenv()

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://172.16.143.229:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# Load The LLM
llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

# Setting Ollama Embeddings
embed_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL
)

# Loading Text
text = """
投保金額:
PD,DPD,MA HK$3,250,000.00
1-2级 HK$3,000,000.00
3-4级 HK$2,000,000.00
5级 HK$1,500,000.00
6级 HK$1,200,000.00
7级-10级 HK$1,000,000.00
"""

# Splitting Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)

# Creating a vector store from Text
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embed_model,
    persist_directory="./chroma_db"
)

# Creating a Retriever
retriever=vector_store.as_retriever()

# Retrieval-QA Chat Pompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Combining Documents (LLM + prompt)
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

# Creating a Retriever Chain
# Retrieve chain is to combining document chain and retrieval chain
# Final Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Invoking the Retrieval Chain
response = retrieval_chain.invoke({
    "input": "What is the insurance amount for grade 3?"
})

# Since diff version and chain setup, the returned dictionary key will not be always "answer"
if "answer" in response:
    print(response['answer'])
if "result" in response:
    print(response['result'])
