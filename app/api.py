from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from typing import Optional, List, Dict

from app.agent import AgentService

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation: Optional[List[Dict]] = None


class ChatResponse(BaseModel):
    reply: str


def get_agent(request: Request) -> AgentService:
    return request.app.state.get_agent_service()


@router.post("/chat/start", response_model=ChatResponse)
async def start_chat(req: ChatRequest, agent: AgentService = Depends(get_agent)):
    reply = await agent.initial_chat(req.message, req.conversation)
    return ChatResponse(reply=reply)


@router.post("/chat/continue", response_model=ChatResponse)
async def continue_chat(req: ChatRequest, agent: AgentService = Depends(get_agent)):
    reply = await agent.chat(req.message)
    return ChatResponse(reply=reply)


@router.post("/chat/reset")
async def reset_chat(agent: AgentService = Depends(get_agent)):
    agent.reset()
    return {"status": "conversation reset"}