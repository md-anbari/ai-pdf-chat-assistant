from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uuid
import os
import logging
import time
from app.core.vector_store import vector_store

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chat_routes")

chat_router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
logger.info("Ensuring 'uploads' directory exists")

@chat_router.post("/process")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    logger.info(f"Received file upload request: {file.filename}")
    
    if not file.filename.endswith(".pdf"):
        logger.error(f"Invalid file format: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate a unique filename
    unique_filename = f"uploads/temp_{uuid.uuid4()}.pdf"
    logger.info(f"Generated unique filename: {unique_filename}")
    
    try:
        # Save the uploaded file
        logger.info(f"Saving uploaded file to {unique_filename}")
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from uploaded file")
        
        with open(unique_filename, "wb") as f:
            f.write(content)
        logger.info(f"File saved successfully to {unique_filename}")
        
        # Record start time for processing
        start_time = time.time()
        
        # Process the PDF
        logger.info(f"Starting PDF processing with vector_store")
        vector_store.process_pdf(unique_filename)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"PDF processing completed successfully in {processing_time:.2f} seconds")
        
        # Clean up the temporary file
        logger.info(f"Cleaning up temporary file: {unique_filename}")
        os.remove(unique_filename)
        logger.info(f"Temporary file removed")
        
        return {"message": "PDF processed successfully", "processing_time": f"{processing_time:.2f} seconds"}
    except Exception as e:
        logger.error(f"Error during PDF processing: {str(e)}", exc_info=True)
        # Clean up in case of error
        if os.path.exists(unique_filename):
            logger.info(f"Cleaning up temporary file after error: {unique_filename}")
            os.remove(unique_filename)
        raise HTTPException(status_code=500, detail=str(e))

@chat_router.post("/chat")
async def chat(chat_request: ChatMessage):
    """Get a response from the assistant."""
    logger.info(f"Received chat request with message: {chat_request.message[:30]}...")
    
    # Measure response time
    start_time = time.time()
    
    try:
        logger.info(f"Sending request to vector_store with {len(chat_request.chat_history)} history items")
        
        # Check if vector store is initialized
        if not hasattr(vector_store, 'vectorstore') or vector_store.vectorstore is None:
            logger.warning("Vector store is not initialized")
            return {
                "response": "Please upload a PDF document first.",
                "sources": []
            }
            
        response, sources = vector_store.get_response(
            chat_request.message,
            chat_request.chat_history
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Got response of length {len(response)} with {len(sources)} sources in {elapsed_time:.2f} seconds")
        
        return {
            "response": response,
            "sources": sources,
            "processing_time": f"{elapsed_time:.2f} seconds"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error during chat (after {elapsed_time:.2f} seconds): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 