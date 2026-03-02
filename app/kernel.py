import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from app.config import settings

from app.config import settings
from app.tools.db_plugin import DBPlugin
from app.repos.base import DBRepository

def create_kernel(repo: DBRepository) -> Kernel:
    # Create kernel
    kernel = Kernel()

    # Add OpenAI chat service
    chat_service = OpenAIChatCompletion(
        ai_model_id=settings.openai_model,
        api_key=settings.openai_api_key,
        service_id="default"
    )
    kernel.add_service(chat_service)

    # Register plugins/tools
    kernel.add_plugin(DBPlugin(repo), plugin_name="db")

    return kernel
