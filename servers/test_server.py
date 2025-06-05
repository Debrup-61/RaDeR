from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
import uvicorn
import argparse
import time
import uuid
from datetime import datetime
import numpy as np

app = FastAPI(title="vLLM Test Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models to match OpenAI API
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "vLLM Test Server is running"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Generate a dummy response
    time.sleep(.2)
    # Generate a dummy response - 70% default response, 30% alternative
    if np.random.random() < 0.7:
        response_message = "This is a dummy response from the test server. I am pretending to be an LLM."
    else:
        response_message = "So the answer is $\\\\boxed{42}$."
    
    # Create the response object
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=response_message
                ),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
    )
    
    return response

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    time.sleep(.2)
    # Process the input to handle both string and list cases
    inputs = request.input if isinstance(request.input, list) else [request.input]
    num_inputs = len(inputs)
    
    # Create dummy embeddings (4096 dimensions per input)
    embeddings = []
    for i, _ in enumerate(inputs):
        # Generate a normalized dummy embedding vector of 4096 dimensions
        embedding = [float(j % 10) / 10 for j in range(4096)]
        
        embeddings.append({
            "object": "embedding",
            "embedding": embedding,
            "index": i
        })
    
    # Estimate token usage (simplified)
    total_tokens = sum(len(text.split()) * 4 for text in inputs)  # Rough estimate
    
    return {
        "object": "list",
        "data": embeddings,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "test-server"
            },
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Run a test vLLM server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on")
    
    args = parser.parse_args()
    
    print(f"Starting vLLM test server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()