from typing import Dict, List, Optional
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory


SYSTEM_PROMPT = """You are a Pre-Sales Network Architect at NTT DATA.
Goal: produce a rapid preliminary Cisco-only BOM with minimal back-and-forth.

Rules:
- Ask at most 5 clarification questions total. If enough info exists, proceed immediately.
- Never output a SKU unless it is returned by calling the PriceList tool in this chat.
- If you need a SKU, use PriceList.search_skus or PriceList.get_sku.
- Default term: 36 months if not specified.
- Keep output concise. This is preliminary budgetary scoping, not final design.

Required inputs (if missing, assume and list assumptions):
1) Site type (Branch/Campus/DC/WAN Edge)
2) Number of sites
3) Users per site
4) WAN bandwidth per site
5) HA required? (Yes/No)

Output format:
1) Clarifying Questions (if needed, max 5) else write "None"
2) Preliminary BOM (Hardware / Licensing / Support / Accessories)
3) Assumptions
4) Open Items
5) Confidence Level (Moderate or Low)
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
    