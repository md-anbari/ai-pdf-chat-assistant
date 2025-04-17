# ai-pdf-chat-assistant

An offline PDF assistant using Retrieval-Augmented Generation (RAG) and LLM models. Designed specifically for processing sensitive documents that should not be shared with online services.

## Technology & Models

- **Backend Framework**: FastAPI 
- **Frontend**: Streamlit - Python library for creating interactive web interfaces with minimal code
- **LLM**: Mistral via Ollama 
- **Embeddings**: mxbai-embed-large via Ollama - High-quality vector embeddings optimized for semantic search
- **Vector Database**: FAISS (Facebook AI Similarity Search) - Efficient similarity search for vector embeddings

## Document Processing Pipeline

1. **PDF Ingestion**: PDF files are loaded using PyPDFLoader, extracting text content from each page
2. **Text Chunking**: Documents are split into smaller chunks (1000 characters with 200 character overlap) using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Each text chunk is converted into a vector embedding (dense numerical representation) using the mxbai-embed-large model
4. **Vector Storage**: Embeddings are stored in a local FAISS vector database for efficient retrieval
5. **Similarity Search**: When a question is asked, the system converts it to an embedding and performs similarity search to find relevant document chunks
6. **Context Preparation**: Retrieved chunks are assembled as context for the LLM
7. **Response Generation**: The Mistral LLM generates responses based on the retrieved document chunks and conversation history

## Why FAISS for Vector Storage?

FAISS was chosen as the vector database for several important reasons:

- **Memory Efficiency**: FAISS is more memory-efficient than alternatives like Chroma DB, making it better suited for local processing of large documents with limited resources
- **Performance**: FAISS implements highly optimized algorithms for similarity search that provide faster query responses
- **Local Processing**: FAISS operates entirely in-memory without requiring external services, maintaining the fully offline nature of the application
- **Simplified Architecture**: No separate server or database process is needed, making the application more self-contained and easier to deploy
- **Lightweight Footprint**: FAISS has minimal dependencies and a smaller resource footprint compared to other vector databases

## Privacy-First Design

- **100% Offline Processing**: All document analysis and AI processing happens locally
- **No Data Sharing**: Documents are never sent to external APIs or cloud services
- **Secure Document Handling**: Ideal for confidential business documents, legal contracts, or personal information that should not be exposed to third-party services

## Project Structure

```
├── app/                   # Backend FastAPI application
│   ├── core/              # Core business logic
│   │   └── vector_store.py  # Vector store implementation 
│   ├── routes/            # API routes
│   │   └── chat.py        # Chat API endpoints
│   └── main.py            # FastAPI application entry point
├── frontend/              # Frontend Streamlit application
│   └── streamlit_app.py   # Streamlit UI
├── uploads/               # Directory for temporary PDF uploads
├── faiss_index/           # Storage for FAISS vector embeddings (created at runtime)
└── requirements.txt       # Project dependencies
```

## Installation and Setup

### Prerequisites

1. Python 3.9+
2. Ollama - [Install Ollama](https://ollama.ai/download)
3. Required Ollama models:
   ```bash
   ollama pull mistral
   ollama pull mxbai-embed-large
   ```

### Create Environment

```bash
# Clone repository
git clone https://github.com/yourusername/smart-pdf-assistant.git
cd smart-pdf-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

1. Start the FastAPI backend:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8005
   ```

2. Start the Streamlit frontend (in a new terminal):
   ```bash
   streamlit run frontend/streamlit_app.py --server.port 8503
   ```

3. Access the application at http://localhost:8503

## Troubleshooting

- **Port Conflicts**: If ports 8005 or 8503 are in use, specify different ports:
  ```bash
  uvicorn app.main:app --reload --host 0.0.0.0 --port [PORT]
  streamlit run frontend/streamlit_app.py --server.port [PORT]
  ```

- **Model Issues**: If embedding or LLM models aren't working, check Ollama:
  ```bash
  ollama list  # Check available models
  ollama pull mistral  # Pull missing models
  ollama pull mxbai-embed-large
  ```