from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, Dict, Any
import os
import json
import asyncio
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIChat

app = FastAPI(title="ICM Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = "openai/gpt-4o"

class ICMResponse(BaseModel):
    conversation_id: str
    message: str
    timestamp: str
    model: str

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ICM_AGENT_PORT = int(os.getenv("ICM_AGENT_PORT", 8001))

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

# Initialize Agno Agent with OpenRouter
def create_agent(model: str = "openai/gpt-4o") -> Agent:
    return Agent(
        model=OpenAIChat(
            id=model,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        ),
        description="ICM Agent for document analysis and intelligent content management",
        instructions="You are an ICM (Intelligent Content Management) agent. Help users analyze documents, extract insights, and manage content effectively."
    )

@app.get("/")
async def root():
    return {"message": "ICM Agent API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/chat", response_model=ICMResponse)
async def chat(request: ChatRequest):
    """
    Standard chat endpoint with synchronous response
    """
    try:
        agent = create_agent(request.model)
        response = agent.run(request.message, stream=False)
        
        return ICMResponse(
            conversation_id=request.conversation_id or f"conv_{datetime.utcnow().timestamp()}",
            message=str(response.content),
            timestamp=datetime.utcnow().isoformat(),
            model=request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with Server-Sent Events (SSE)
    """
    return StreamingResponse(
        generate_stream_response(request),
        media_type="text/event-stream"
    )

async def generate_stream_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Generate streaming response with SSE format
    """
    conversation_id = request.conversation_id or f"conv_{datetime.utcnow().timestamp()}"
    
    try:
        agent = create_agent(request.model)
        
        # Send initial connection message
        yield f"data: {json.dumps({'status': 'connected', 'conversation_id': conversation_id})}\n\n"
        
        # Run the agent with streaming
        response_stream = agent.run(request.message, stream=True)
        
        buffer = ""
        async for chunk in response_stream:
            if chunk.content:
                buffer += str(chunk.content)
                # Send each chunk as it arrives
                yield f"data: {json.dumps({
                    'type': 'chunk',
                    'content': str(chunk.content),
                    'conversation_id': conversation_id
                })}\n\n"
        
        # Send completion message
        yield f"data: {json.dumps({
            'type': 'complete',
            'content': buffer,
            'conversation_id': conversation_id,
            'timestamp': datetime.utcnow().isoformat(),
            'model': request.model
        })}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

@app.get("/api/models")
async def list_models():
    """
    List available models from OpenRouter
    """
    return {
        "models": [
            {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI"},
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus", "provider": "Anthropic"},
            {"id": "anthropic/claude-3-sonnet", "name": "Claude 3 Sonnet", "provider": "Anthropic"},
            {"id": "google/gemini-pro", "name": "Gemini Pro", "provider": "Google"}
        ]
    }

@app.post("/api/analyze-document")
async def analyze_document(request: Request):
    """
    Document analysis endpoint
    """
    try:
        body = await request.json()
        document_content = body.get("content", "")
        
        if not document_content:
            raise HTTPException(status_code=400, detail="Document content is required")
        
        agent = create_agent()
        prompt = f"Please analyze the following document and provide insights:\n\n{document_content}"
        response = agent.run(prompt, stream=False)
        
        return {
            "analysis": str(response.content),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ICM_AGENT_PORT)