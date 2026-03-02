import asyncio
from app.kernel import create_kernel
from app.repos.memory import InMemoryRepository
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

async def main():
    repo = InMemoryRepository()
    kernel = await create_kernel(repo)

    chat_service = kernel.get_service(service_id="chat_main")
    execution_settings = OpenAIChatPromptExecutionSettings()
    chat_history = ChatHistory()
    chat_history.add_user_message("Hello, how are you?")

    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
    )

    print("Response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())