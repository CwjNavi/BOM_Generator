from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import uuid4

from app.agent import AgentService

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # optional: seed conversation messages on start
    conversation: Optional[List[Dict]] = None
    # If not provided on /chat/start, we generate one and return it
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    stage: Optional[str] = None
    requirements: Optional[Dict] = None
    missing: Optional[List[str]] = None


def get_agent(request: Request) -> AgentService:
    return request.app.state.get_agent_service()


@router.post("/chat/start", response_model=ChatResponse)
async def start_chat(req: ChatRequest, agent: AgentService = Depends(get_agent)):
    """
    Start a new conversation. If conversation_id is not provided, generate one.
    """
    conversation_id = req.conversation_id or str(uuid4())

    # If your AgentService supports a per-conversation reset/start, call it here.
    try:
        reply = await agent.initial_chat(
            conversation_id=conversation_id,
            message=req.message,
            conversation=req.conversation,
        )
        meta = agent.get_conversation_meta(conversation_id)  # optional helper
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="AgentService must support initial_chat(conversation_id=..., ...)",
        )

    return ChatResponse(
        reply=reply,
        conversation_id=conversation_id,
        stage=meta.get("stage"),
        requirements=meta.get("requirements"),
        missing=meta.get("missing"),
    )


@router.post("/chat/continue", response_model=ChatResponse)
async def continue_chat(req: ChatRequest, agent: AgentService = Depends(get_agent)):
    """
    Continue an existing conversation.
    """
    if not req.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required for /chat/continue")

    try:
        reply = await agent.chat(conversation_id=req.conversation_id, message=req.message)
        meta = agent.get_conversation_meta(req.conversation_id)  # optional helper
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown conversation_id. Call /chat/start first.")
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="AgentService must support chat(conversation_id=..., message=...)",
        )

    return ChatResponse(
        reply=reply,
        conversation_id=req.conversation_id,
        stage=meta.get("stage"),
        requirements=meta.get("requirements"),
        missing=meta.get("missing"),
    )


@router.post("/chat/reset/{conversation_id}")
async def reset_chat(conversation_id: str, agent: AgentService = Depends(get_agent)):
    """
    Reset a specific conversation.
    """
    try:
        agent.reset(conversation_id=conversation_id)
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="AgentService must support reset(conversation_id=...)",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown conversation_id")

    return {"status": "conversation reset", "conversation_id": conversation_id}


# Optional: convenience endpoint that starts if conversation_id absent, continues otherwise
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, agent: AgentService = Depends(get_agent)):
    """
    Single endpoint:
    - If conversation_id provided => continue
    - Else => start a new conversation and return generated conversation_id
    """
    if req.conversation_id:
        reply = await agent.chat(conversation_id=req.conversation_id, message=req.message)
        meta = agent.get_conversation_meta(req.conversation_id)
        return ChatResponse(
            reply=reply,
            conversation_id=req.conversation_id,
            stage=meta.get("stage"),
            requirements=meta.get("requirements"),
            missing=meta.get("missing"),
        )

    conversation_id = str(uuid4())
    reply = await agent.initial_chat(
        conversation_id=conversation_id,
        message=req.message,
        conversation=req.conversation,
    )
    meta = agent.get_conversation_meta(conversation_id)
    return ChatResponse(
        reply=reply,
        conversation_id=conversation_id,
        stage=meta.get("stage"),
        requirements=meta.get("requirements"),
        missing=meta.get("missing"),
    )