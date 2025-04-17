from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import shutil
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Tuple, Dict
import logging

# NOTE: This is the primary VectorStore implementation used by the app.
# The vector.py file in the project root is a legacy implementation that
# is no longer used and can be safely deleted to avoid confusion.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vector_store")

# Define memory window size - how many conversation turns to keep
# This prevents context overflow in large conversations
MEMORY_WINDOW_SIZE = 5

class VectorStore:
    """
    Manages document embeddings storage and retrieval using FAISS vector database.
    
    This class handles PDF processing, text chunking, embedding generation,
    and conversational retrieval of information from documents using the FAISS
    vector database for efficient similarity search.
    """
    
    def __init__(self, faiss_index_folder="faiss_index"):
        """
        Initialize vector store with embeddings and text processing components.
        
        Args:
            faiss_index_folder (str): Directory where FAISS index will be stored if saving to disk
        """
        logger.info("Initializing VectorStore")
        self.faiss_index_folder = faiss_index_folder
        
        # Initialize embedding model for converting text to vector representations
        logger.info(f"Using embedding model: mxbai-embed-large")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vectorstore = None
        
        # Initialize language model for generating responses
        logger.info(f"Using LLM model: mistral")
        self.llm = Ollama(model="mistral")
        
        # Configure text splitter for breaking documents into manageable chunks
        # Chunk size determines the context window for each embedding
        # Overlap ensures continuity between chunks to avoid losing context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Use windowed memory to limit context size while maintaining conversation flow
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=MEMORY_WINDOW_SIZE,  # Keep only last N conversation turns
            return_messages=True
        )
        logger.info(f"Using ConversationBufferWindowMemory with window size: {MEMORY_WINDOW_SIZE}")
        self.qa_chain = None
        logger.info("VectorStore initialization complete")

    def add_document(self, file_path: str):
        """
        Add a PDF document to the vector store.
        
        This method loads the PDF, splits it into chunks, creates embeddings,
        and adds them to the FAISS vector store.
        
        Args:
            file_path (str): Path to the PDF file to be processed
        """
        logger.info(f"Adding document: {file_path}")
        
        # Load document using PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info(f"Loaded {len(pages)} pages from PDF")
        
        # Split document into manageable text chunks
        texts = self.text_splitter.split_documents(pages)
        logger.info(f"Split into {len(texts)} text chunks")
        
        # Either create a new vector store or add to existing one
        if self.vectorstore is None:
            logger.info("Creating new FAISS vector store")
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        else:
            logger.info("Adding documents to existing FAISS vector store")
            self.vectorstore.add_documents(texts)
        logger.info("Document added successfully")

    def similarity_search(self, query: str, k: int = 3):
        """
        Search for similar documents in the vector store using semantic similarity.
        
        Converts the query to an embedding and finds the most similar document chunks.
        
        Args:
            query (str): The search query text
            k (int): Number of similar documents to retrieve (default: 3)
            
        Returns:
            list: List of document chunks that match the query
        """
        logger.info(f"Performing similarity search for query: {query[:30]}...")
        if self.vectorstore is None:
            logger.warning("Vector store is None, returning empty list")
            return []
        
        # Perform vector similarity search
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} similar documents")
        return results

    def load_pdf(self, pdf_path):
        """
        Load PDF, split into chunks, and create embeddings in a new FAISS vector store.
        
        This method replaces any existing vector store with a new one.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            FAISS: The created vector store
        """
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Clean up existing vector store if present
        if os.path.exists(self.faiss_index_folder):
            logger.info(f"Removing existing FAISS index directory: {self.faiss_index_folder}")
            shutil.rmtree(self.faiss_index_folder)
        
        # Load and process PDF document
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logger.info(f"Number of pages loaded: {len(pages)}")
        
        # Split document into chunks for embedding
        texts = self.text_splitter.split_documents(pages)
        logger.info(f"Number of text chunks created: {len(texts)}")
        
        # Create vector store using FAISS
        logger.info("Creating FAISS vector store from documents")
        vectorstore = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
        logger.info(f"FAISS vector store created with {len(texts)} documents")
        self.vectorstore = vectorstore
        return vectorstore

    def get_retriever(self, k=3):
        """
        Get a retriever for searching documents from the FAISS vector store.
        
        The retriever provides a standardized interface for document retrieval.
        
        Args:
            k (int): Number of documents to retrieve per search (default: 3)
            
        Returns:
            Retriever: A retriever object or None if index directory doesn't exist
        """
        logger.info(f"Getting retriever with k={k}")
        if not os.path.exists(self.faiss_index_folder):
            logger.warning(f"FAISS index directory {self.faiss_index_folder} does not exist")
            return None
        
        # Create vectorstore from FAISS index
        vectorstore = FAISS.load_local(
            folder_path=self.faiss_index_folder,
            embeddings=self.embeddings
        )
        logger.info("Created retriever from FAISS vector store")
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def process_pdf(self, pdf_path: str) -> None:
        """
        Process a PDF file and create the necessary components for question answering.
        
        This method handles the complete pipeline:
        1. Loading the PDF
        2. Splitting into chunks
        3. Creating embeddings in FAISS
        4. Setting up the QA chain
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: For other processing errors
        """
        try:
            logger.info(f"Starting to process PDF: {pdf_path}")
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Load PDF document
            logger.info(f"Loading PDF with PyPDFLoader")
            loader = PyPDFLoader(pdf_path)
            
            logger.info(f"Reading PDF content")
            pages = loader.load()
            logger.info(f"Successfully loaded {len(pages)} pages from PDF")
            
            # Split document into chunks with overlap to maintain context
            logger.info(f"Splitting text into chunks with chunk_size=1000, overlap=200")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            logger.info(f"Created {len(chunks)} text chunks from PDF")
            
            # Create vector store with FAISS from document chunks
            logger.info(f"Creating FAISS vector store with {self.embeddings.__class__.__name__}")
            try:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                logger.info(f"Successfully created FAISS vector store with {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error creating FAISS vector store: {str(e)}")
                raise
            
            # Reset memory when loading a new document to start fresh conversation
            self.memory.clear()
            logger.info("Cleared conversation memory")
            
            # Create QA chain combining the language model, retriever, and memory
            logger.info(f"Creating ConversationalRetrievalChain with model: {self.llm.__class__.__name__}")
            try:
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore.as_retriever(),
                    memory=self.memory
                )
                logger.info("Successfully created QA chain")
            except Exception as e:
                logger.error(f"Error creating QA chain: {str(e)}")
                raise
            
            logger.info("PDF processing completed successfully")
        except Exception as e:
            logger.error(f"Error in process_pdf: {str(e)}", exc_info=True)
            raise

    def get_response(self, query: str, chat_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        """
        Get a response from the QA chain based on the query and chat history.
        
        This method:
        1. Handles the conversation context using chat history
        2. Queries the language model with relevant document chunks
        3. Retrieves source documents for citation
        
        Args:
            query (str): The user's question
            chat_history (List[Dict]): Previous conversation messages in role/content format
            
        Returns:
            Tuple[str, List[Dict]]: A tuple containing (response text, source documents)
            
        Raises:
            Exception: For errors during response generation
        """
        try:
            logger.info(f"Getting response for query: {query[:30]}...")
            
            # Check if QA chain is initialized
            if not self.qa_chain:
                logger.warning("QA chain not initialized. Asking user to upload PDF first.")
                return "Please upload a PDF first.", []
            
            # Process chat history if provided, maintaining conversation context
            if chat_history:
                # Only use the most recent messages if there are too many
                max_history = min(len(chat_history), MEMORY_WINDOW_SIZE * 2)  # *2 because each turn has user+assistant
                recent_history = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
                
                logger.info(f"Processing chat history with {len(recent_history)} messages (limited from {len(chat_history)})")
                # Clear existing memory to avoid duplication
                self.memory.clear()
                
                # Add conversation turns to memory
                for msg in recent_history:
                    if msg["role"] == "user":
                        self.memory.chat_memory.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        self.memory.chat_memory.add_ai_message(msg["content"])
            
            # Query the QA chain with the user's question
            logger.info("Querying QA chain")
            result = self.qa_chain({"question": query})
            response = result["answer"]
            logger.info(f"Got response of length {len(response)}")
            
            # Retrieve source documents for citation
            sources = []
            if self.vectorstore:
                logger.info("Retrieving source documents")
                docs = self.vectorstore.similarity_search(query, k=3)
                logger.info(f"Found {len(docs)} source documents")
                for doc in docs:
                    sources.append({
                        "page": doc.metadata.get("page", 0),
                        "content": doc.page_content
                    })
            
            logger.info("Successfully generated response with sources")
            return response, sources
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            raise

# Create a global instance of VectorStore for use throughout the application
logger.info("Creating global VectorStore instance")
vector_store = VectorStore()
logger.info("Global VectorStore instance created") 