"""
NOTE: This file is redundant and not currently used in the application.
The functionality provided by LLMProcessor has been incorporated into the
VectorStore class in app/core/vector_store.py.

This file can be safely deleted to avoid confusion and maintenance overhead.
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from typing import List, Tuple

class LLMProcessor:
    def __init__(self):
        self.llm = Ollama(model="mistral")
        self.chain = self._initialize_chain()
        self.chat_history: List[Tuple[str, str]] = []
        
    def _initialize_chain(self):
        template = """
        You are a helpful AI assistant for analyzing PDF documents. 
        Use the following context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Previous conversation:
        {chat_history}
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def format_chat_history(self):
        return "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in self.chat_history])
    
    def process_query(self, question: str, context_docs: List) -> str:
        # Format the chat history
        formatted_history = self.format_chat_history()
        
        # Get the answer
        response = self.chain.invoke({
            "context": self.format_docs(context_docs),
            "question": question,
            "chat_history": formatted_history
        })
        
        # Add to chat history
        self.chat_history.append((question, response))
        
        return response
    
    def clear_history(self):
        self.chat_history = [] 