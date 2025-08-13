# RAG Implementation with Ollama and LangChain

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system using Ollama for local LLM inference and LangChain for orchestrating the RAG pipeline.

## 🔍 Overview

RAG combines the power of document retrieval with language model generation to provide accurate, context-aware answers based on your own data. This implementation:

1. **Embeds documents** into a vector database for semantic search
2. **Retrieves relevant chunks** based on user queries  
3. **Combines retrieved context** with prompts for the LLM
4. **Generates grounded answers** using only the provided context

## 🏗️ Architecture

```
User Query → Retriever → Document Chunks → LLM + Prompt → Final Answer
```

The system follows this flow:
- **Embed data** into vector DB → **Retriever** finds relevant chunks → **Combine** answer with prompt → **Retrieval chain** delivers the final answer

## 📋 Prerequisites

- **Python 3.8+**
- **Ollama** installed and running locally
- **LangChain** libraries

### Required Python packages:
```bash
pip install langchain langchain-ollama python-dotenv chromadb
```

## 🚀 Setup

1. **Install and start Ollama:**
   ```bash
   # Install Ollama (visit https://ollama.ai for instructions)
   ollama pull llama3.2:latest  # or your preferred model
   ollama serve  # Start the Ollama server
   ```

2. **Environment Configuration:**
   Create a `.env` file in your project root:
   ```env
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:latest
   ```

3. **Run the script:**
   ```bash
   python rag_demo.py
   ```

## 🎯 Features

- **Local LLM Processing**: Uses Ollama for privacy-preserving, local inference
- **Vector Search**: Implements semantic search using Chroma vector database  
- **Chinese/English Support**: Handles multilingual content (demonstrated with insurance data)
- **Persistent Storage**: Saves embeddings to `./chroma_db` for reuse
- **Flexible Output**: Handles different response formats across LangChain versions

## 💡 Code Structure

### Key Components:

1. **Document Processing**:
   ```python
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
   chunks = text_splitter.split_text(text)
   ```

2. **Vector Store Creation**:
   ```python
   vector_store = Chroma.from_texts(
       texts=chunks,
       embedding=embed_model,
       persist_directory="./chroma_db"
   )
   ```

3. **RAG Pipeline**:
   ```python
   retriever = vector_store.as_retriever()
   combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
   retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
   ```

## 📊 Example Usage

The demo includes insurance coverage data in Chinese and responds to queries like:

**Query**: "What is the insurance amount for grade 3?"  
**Response**: "For grade 3-4, the insurance amount is HK$2,000,000.00"

## ⚙️ Configuration Options

- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` for different document types
- **Model Selection**: Change `OLLAMA_MODEL` to use different Ollama models
- **Retriever Settings**: Modify retriever parameters for different search behaviors

## 🐛 Troubleshooting

### Common Issues:

1. **404 Errors**: Ensure Ollama is running and accessible at the specified URL
2. **Model Not Found**: Pull the required model with `ollama pull <model_name>`  
3. **Embedding Issues**: Verify your model supports embeddings (use `nomic-embed-text` for embeddings if needed)

### Version Compatibility:
The code handles different LangChain versions by checking for both `"answer"` and `"result"` keys in the response.

## 🔧 Customization

To use your own data:
1. Replace the `text` variable with your content
2. Adjust chunk parameters based on your document structure  
3. Modify the query example to match your use case

## 📈 Performance Tips

- **Persistent Vector Store**: The embeddings are saved locally, so subsequent runs will be faster
- **Model Selection**: Choose appropriate models based on your hardware capabilities
- **Chunk Optimization**: Experiment with different chunk sizes for your specific content

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).