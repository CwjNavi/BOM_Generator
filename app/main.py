from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router
from app.repos.memory import InMemoryRepository
from app.kernel import create_kernel
from app.agent import AgentService

from app.agents import quotation_agent, extractor_agent, validation_agent, generator_agent


def create_app() -> FastAPI:
    app = FastAPI(title="bom generator")

    repo = InMemoryRepository()
    kernel = create_kernel(repo)

    agent = AgentService(
        kernel=kernel,
        quotation_agent=quotation_agent,
        extractor_agent=extractor_agent,
        validation_agent=validation_agent,
        generator_agent=generator_agent,
        service_id="default",
    )

    def get_agent_service() -> AgentService:
        return agent

    app.state.get_agent_service = get_agent_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/")
    async def base():
        return {"hello": "world"}

    return app


app = create_app()