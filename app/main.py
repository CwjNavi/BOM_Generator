from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from app.api import router as api_router

from app.repos.memory import InMemoryRepository
from app.kernel import create_kernel
from app.agent import AgentService


def create_app() -> FastAPI:
    app = FastAPI(title="bom generator")

    # ---------- Infrastructure Wiring ----------
    repo = InMemoryRepository()
    kernel = create_kernel(repo)
    agent = AgentService(kernel)

    # ---------- Dependency Injection ----------
    def get_agent_service() -> AgentService:
        return agent

    # attach dependency to app state (cleaner than globals)
    app.state.get_agent_service = get_agent_service

    # ---------- Middleware ----------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- Routers ----------
    app.include_router(api_router)

    # ---------- Health ----------
    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/")
    async def base():
        return {"hello": "world"}
    
    return app

app = create_app()