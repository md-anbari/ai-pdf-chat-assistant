from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat import chat_router

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Smart PDF Assistant API"} 