from typing import Dict, List, Optional
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory


SYSTEM_PROMPT = """You are an expert Cisco networking hardware consultant. Generate a detailed Bill of Materials (BOM) based on the customer requirements provided.

Rules:
1. Only include genuine Cisco products with valid SKU/part numbers
2. Include all necessary components (main units, power supplies, licenses, support contracts)
3. Provide realistic quantities based on the requirements
4. Include brief descriptions for each item
5. Format as a clear table

Output format:
1. First, provide a brief summary of the solution
2. Then, list the BOM as a markdown table with columns: SKU, Description, Quantity
3. Finally, provide any implementation notes or recommendations

Requirements gathered:
""".strip()

class AgentService:
    def __init__(self, kernel: sk.Kernel) -> None:
        self._kernel = kernel
        self._chat_history: ChatHistory | None = None

    async def initial_chat(
        self,
        message: str,
        conversation: Optional[List[Dict]] = None,
    ) -> str:
        """
        Starts a brand new conversation.
        Resets internal chat history.
        """
        # Create fresh history
        self._chat_history = ChatHistory()
        self._chat_history.add_system_message(SYSTEM_PROMPT)

        # Optional prior conversation injection
        if conversation:
            for m in conversation:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    self._chat_history.add_user_message(content)
                elif role == "assistant":
                    self._chat_history.add_assistant_message(content)

        # Add latest message
        self._chat_history.add_user_message(message)

        return await self._invoke_model()

    async def chat(self, message: str) -> str:
        """
        Continues an existing conversation using stored chat history.
        Raises error if conversation not initialized.
        """
        if self._chat_history is None:
            raise RuntimeError(
                "Conversation not initialized. Call initial_chat() first."
            )

        self._chat_history.add_user_message(message)

        return await self._invoke_model()

    async def _invoke_model(self) -> str:
        """
        Internal helper to call the model using current chat history.
        """
        execution_settings = self._kernel.get_prompt_execution_settings_from_service_id("default")

        chat_service = self._kernel.get_service(service_id="default")

        response = await chat_service.get_chat_message_content(
            chat_history=self._chat_history,
            settings=execution_settings,
        )

        # Add assistant reply to history for continuity
        self._chat_history.add_assistant_message(str(response))

        return str(response)

    def reset(self):
        """
        Manually reset conversation state.
        """
        self._chat_history = None
    