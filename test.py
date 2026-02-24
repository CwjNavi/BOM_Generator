import asyncio
import os
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

load_dotenv()

async def main():
    # Create kernel
    kernel = Kernel()

    # Add OpenAI chat service
    chat_service = OpenAIChatCompletion(
        ai_model_id=os.environ["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"],
        api_key=os.environ["OPENAI_API_KEY"]
    )
    kernel.add_service(chat_service)

    # Settings
    execution_settings = OpenAIChatPromptExecutionSettings()

    # Chat history
    chat_history = ChatHistory()
    chat_history.add_user_message("Hello, how are you?")

    # Call model
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
    )

    print(response)

asyncio.run(main())