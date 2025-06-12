import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Please set the GROQ_API_KEY in the .env file.")

app = FastAPI()

# Serve static files (HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]
        self.active: bool = True

conversations: Dict[str, Conversation] = {}

def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]

@app.get("/", response_class=HTMLResponse)
async def get_chat_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(
            status_code=400,
            detail="The chat session has ended. Please start a new session.",
        )

    try:
        conversation.messages.append({
            "role": input.role,
            "content": input.message
        })

        response = query_groq_api(conversation)

        conversation.messages.append({
            "role": "assistant",
            "content": response
        })

        return {
            "response": response,
            "conversation_id": input.conversation_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
