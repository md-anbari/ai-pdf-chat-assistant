import streamlit as st
import requests
import json
import time
import random
import threading
from typing import List, Dict
import os

# Set page config
st.set_page_config(
    page_title="Smart PDF Assistant",
    page_icon="📚",
    layout="wide"
)

# Define max history length to prevent token limit issues
MAX_HISTORY_LENGTH = 10

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "new_question" not in st.session_state:
    st.session_state.new_question = None
if "animation_thread" not in st.session_state:
    st.session_state.animation_thread = None

# API endpoint
API_ENDPOINT = "http://localhost:8005/api/v1"

def handle_submit():
    """Callback for when user submits a question"""
    prompt = st.session_state.user_input
    if prompt:
        # Store the new question in session state
        st.session_state.new_question = prompt
        # Set waiting flag
        st.session_state.waiting_for_response = True
        # Don't try to clear input box directly - not allowed in Streamlit

def display_chat_message(message: Dict) -> None:
    """Display a chat message in the chat interface."""
    with st.chat_message(message["role"]):
        st.write(message["content"])

def simulate_stream_response(response: str) -> str:
    """Simulate a streaming response with a smoother, more natural text animation."""
    message_placeholder = st.empty()
    full_response = ""
    
    # Process the text one word at a time for a smoother effect
    words = response.split()
    
    # Group words into small batches for smoother animation
    batch_size = 2  # Number of words per batch
    for i in range(0, len(words), batch_size):
        # Get the current batch of words
        batch = words[i:i+batch_size]
        word_group = " ".join(batch)
        
        # Add to full response
        full_response += word_group + " "
        message_placeholder.write(full_response)
        
        # Vary the speed slightly for more natural feel
        delay = 0.05 + (0.05 * random.random())  # 0.05-0.1 seconds
        time.sleep(delay)
        
        # Add occasional longer pauses at punctuation
        if any(p in word_group for p in ['.', '?', '!']):
            time.sleep(0.2)
    
    return full_response

def animate_thinking():
    """Display an animated thinking indicator."""
    thinking_placeholder = st.empty()
    
    # Use container to keep the same space for different length messages
    with thinking_placeholder.container():
        # The thinking animation frames
        frames = ["Thinking.", "Thinking..", "Thinking..."]
        
        for _ in range(15):  # Run for 15 cycles or until interrupted
            for frame in frames:
                thinking_placeholder.markdown(f"*{frame}*")
                time.sleep(0.3)
                # Check if we should stop the animation
                if not st.session_state.get("waiting_for_response", True):
                    break
            # If not waiting anymore, stop animating
            if not st.session_state.get("waiting_for_response", True):
                break
                
    return thinking_placeholder

def display_sources(sources: List[Dict]) -> None:
    """Display the sources for the last response."""
    if sources:
        with st.expander("Sources"):
            for i, source in enumerate(sources, 1):
                st.write(f"**Source {i} (Page {source['page'] + 1})**")
                st.write(source["content"])
                st.divider()

def trim_chat_history(history: List[Dict], max_length: int) -> List[Dict]:
    """Trim chat history to keep only the most recent messages."""
    if len(history) <= max_length:
        return history
    
    # Always keep the most recent messages
    return history[-max_length:]

# Sidebar for PDF upload
with st.sidebar:
    st.title("📚 Smart PDF Assistant")
    st.write("Upload a PDF document to start chatting")
    
    # Status area for processing steps
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
        
    status_placeholder = st.empty()
    
    # Only show PDF uploader if we're not in the middle of a chat
    if not st.session_state.waiting_for_response:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file and not st.session_state.pdf_processed:
            # Create a status area with an expander
            with status_placeholder.container():
                status_area = st.expander("Processing Status", expanded=True)
                with status_area:
                    st.markdown("### PDF Processing Steps")
                    
                    # Initialize progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Saving file
                    status_text.markdown("⏳ **Step 1/5:** Saving uploaded file...")
                    # Save the uploaded file
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    progress_bar.progress(20)
                    status_text.markdown("✅ **Step 1/5:** File saved successfully!")
                    
                    # Step 2: Sending to API
                    status_text.markdown("⏳ **Step 2/5:** Sending to processing API...")
                    progress_bar.progress(30)
                    
                    # Process the PDF
                    try:
                        # Step 3: Processing on server
                        response = requests.post(
                            f"{API_ENDPOINT}/process",
                            files={"file": ("temp.pdf", open("temp.pdf", "rb"), "application/pdf")}
                        )
                        
                        if response.status_code == 200:
                            # Step 4: Creating vector store
                            progress_bar.progress(50)
                            status_text.markdown("✅ **Step 2/5:** File sent to server!")
                            status_text.markdown("⏳ **Step 3/5:** Extracting text from PDF...")
                            time.sleep(0.5)  # Add small delay for visual feedback
                            
                            progress_bar.progress(70)
                            status_text.markdown("✅ **Step 3/5:** Text extracted successfully!")
                            status_text.markdown("⏳ **Step 4/5:** Creating vector embeddings...")
                            time.sleep(0.5)  # Add small delay for visual feedback
                            
                            progress_bar.progress(90)
                            status_text.markdown("✅ **Step 4/5:** Vector embeddings created!")
                            status_text.markdown("⏳ **Step 5/5:** Finalizing...")
                            time.sleep(0.5)  # Add small delay for visual feedback
                            
                            progress_bar.progress(100)
                            status_text.markdown("✅ **Step 5/5:** Processing complete!")
                            
                            # Show success message
                            st.success("PDF processed successfully and ready for questions!")
                            st.session_state.pdf_processed = True
                        else:
                            progress_bar.progress(100)
                            status_text.markdown("❌ **Error:** Failed to process PDF")
                            st.error(f"Error processing PDF: {response.text}")
                    except Exception as e:
                        progress_bar.progress(100)
                        status_text.markdown(f"❌ **Error:** {str(e)}")
                        st.error(f"Error: {str(e)}")
                    finally:
                        # Clean up
                        if os.path.exists("temp.pdf"):
                            os.remove("temp.pdf")
    
    # Show file information if processed
    if st.session_state.pdf_processed:
        st.success("PDF loaded and ready for questions!")
        if st.button("Reset and Upload New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.sources = []
            st.session_state.waiting_for_response = False
            st.session_state.new_question = None
            st.rerun()

# Main chat interface
st.title("Chat with your PDF")

# Reset the interface completely if needed
if st.button("Start New Chat", type="primary"):
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.session_state.waiting_for_response = False
    st.session_state.new_question = None
    st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    display_chat_message(message)
    
    # Display sources after assistant messages
    if message["role"] == "assistant" and st.session_state.sources:
        display_sources(st.session_state.sources)
        # Clear sources after displaying to avoid duplication
        st.session_state.sources = []

# Process any new question in session state
if st.session_state.new_question:
    # Get the question
    prompt = st.session_state.new_question
    
    # Add user message to chat history
    user_message = {"role": "user", "content": prompt}
    st.session_state.chat_history.append(user_message)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Clear the question
    st.session_state.new_question = None
    
    # Create an AI message slot
    with st.chat_message("assistant"):
        thinking_container = st.empty()
        
        # Get response from API
        try:
            if not st.session_state.pdf_processed:
                st.warning("Please upload a PDF document first.")
                st.session_state.waiting_for_response = False
                st.stop()
                
            # Show a simple spinner with text
            with thinking_container:
                with st.spinner("Thinking..."):
                    # Trim chat history to prevent token limit issues
                    limited_history = trim_chat_history(st.session_state.chat_history[:-1], MAX_HISTORY_LENGTH-1)
                    
                    # Make API call
                    response = requests.post(
                        f"{API_ENDPOINT}/chat",
                        json={
                            "message": prompt,
                            "chat_history": limited_history  # Send limited history to API
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
            
            # Clear container after getting response
            thinking_container.empty()
            
            # If we got a successful response, display it
            if response.status_code == 200:
                # Simulate streaming response
                assistant_response = simulate_stream_response(data["response"])
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                
                # Store sources for display on next refresh
                st.session_state.sources = data["sources"]
                
                # Trim chat history again after adding assistant response
                st.session_state.chat_history = trim_chat_history(st.session_state.chat_history, MAX_HISTORY_LENGTH)
            else:
                st.error(f"Error getting response from the API: {response.text}")
        except Exception as e:
            thinking_container.empty()
            st.error(f"Error: {str(e)}")
        
        # Clear waiting flag
        st.session_state.waiting_for_response = False

# Chat input - only show when not waiting for a response
if not st.session_state.waiting_for_response:
    st.chat_input(
        placeholder="Ask a question about the PDF",
        key="user_input", 
        on_submit=handle_submit
    )
else:
    # Placeholder to keep UI consistent
    st.text("")

# Show a message if no PDF has been uploaded
if not st.session_state.pdf_processed and not st.session_state.chat_history:
    st.info("👈 Please upload a PDF document using the sidebar to start chatting.") 