from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Tuple

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory


# ---------------------------
# Conversation stage machine
# ---------------------------
class ConversationStage(Enum):
    COLLECTING_REQUIREMENTS = 1
    VALIDATING = 2
    AWAITING_CONFIRMATION = 3
    GENERATING_FINAL_BOM = 4
    COMPLETE = 5


# ---------------------------
# Requirements state schema
# ---------------------------
class RequirementItem(TypedDict, total=False):
    product_name: str
    sku: Optional[str]  # None if unknown/unconfirmed
    quantity: int


class RequirementsState(TypedDict):
    items: List[RequirementItem]


@dataclass
class ConversationState:
    stage: ConversationStage = ConversationStage.COLLECTING_REQUIREMENTS
    requirements: RequirementsState = field(default_factory=lambda: {"items": []})
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    last_validation: Dict[str, Any] = field(default_factory=dict)


REQUIREMENTS_MARKER = "##CURRENT_REQUIREMENTS_JSON##"


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None


def append_requirements_snapshot(history: ChatHistory, requirements: RequirementsState) -> None:
    history.add_system_message(
        f"{REQUIREMENTS_MARKER}\n{json.dumps(requirements, ensure_ascii=False, indent=2)}"
    )


def _looks_like_confirmation(user_text: str) -> bool:
    t = user_text.strip().lower()
    confirmations = ("yes", "yep", "yeah", "correct", "confirm", "confirmed", "looks good", "go ahead", "proceed")
    return any(p in t for p in confirmations)


# ---------------------------
# AgentService (multi-conversation)
# ---------------------------
class AgentService:
    """
    Multi-conversation AgentService.

    Stores per-conversation:
      - ChatHistory
      - ConversationState
      - asyncio.Lock (prevents interleaving writes on concurrent requests)
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        quotation_agent: ChatCompletionAgent,
        extractor_agent: ChatCompletionAgent,
        validation_agent: ChatCompletionAgent,
        generator_agent: ChatCompletionAgent,
        *,
        service_id: str = "default",
    ) -> None:
        self._kernel = kernel
        self._service_id = service_id

        self._quotation_agent = quotation_agent
        self._extractor_agent = extractor_agent
        self._validation_agent = validation_agent
        self._generator_agent = generator_agent

        # conversation_id -> (ChatHistory, ConversationState)
        self._conversations: Dict[str, Tuple[ChatHistory, ConversationState]] = {}

        # conversation_id -> asyncio.Lock
        self._locks: Dict[str, asyncio.Lock] = {}

        # lock protecting dictionaries above
        self._registry_lock = asyncio.Lock()

    # -------- Public API --------

    async def initial_chat(
        self,
        *,
        conversation_id: str,
        message: str,
        conversation: Optional[List[Dict]] = None,
    ) -> str:
        """
        Starts (or restarts) a conversation for a conversation_id.
        """
        async with self._registry_lock:
            history = ChatHistory()
            state = ConversationState()
            self._conversations[conversation_id] = (history, state)
            self._locks.setdefault(conversation_id, asyncio.Lock())

        # Fill injected conversation AFTER registering (so /continue can’t race)
        if conversation:
            for m in conversation:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    history.add_user_message(content)
                elif role == "assistant":
                    history.add_assistant_message(content)

        # Seed requirements snapshot
        append_requirements_snapshot(history, state.requirements)

        # Now handle the first message
        return await self.chat(conversation_id=conversation_id, message=message)

    async def chat(self, *, conversation_id: str, message: str) -> str:
        """
        Continue a conversation by conversation_id.
        """
        history, state = self._get_conversation(conversation_id)

        # Ensure messages don’t interleave for same conversation_id
        lock = await self._get_lock(conversation_id)
        async with lock:
            history.add_user_message(message)

            if state.stage == ConversationStage.COLLECTING_REQUIREMENTS:
                return await self._handle_collecting_requirements(history, state, message)

            if state.stage == ConversationStage.VALIDATING:
                return await self._handle_validating(history, state)

            if state.stage == ConversationStage.AWAITING_CONFIRMATION:
                return await self._handle_awaiting_confirmation(history, state, message)

            if state.stage == ConversationStage.GENERATING_FINAL_BOM:
                return await self._handle_generating_final_bom(history, state)

            history.add_assistant_message(
                "This BOM workflow is complete. If you want a new quote, start a new conversation."
            )
            return "This BOM workflow is complete. If you want a new quote, start a new conversation."

    def reset(self, *, conversation_id: Optional[str] = None) -> None:
        """
        Reset one conversation, or all if conversation_id is None.
        """
        if conversation_id is None:
            self._conversations.clear()
            self._locks.clear()
            return

        if conversation_id not in self._conversations:
            raise KeyError("Unknown conversation_id")

        self._conversations.pop(conversation_id, None)
        self._locks.pop(conversation_id, None)

    def get_conversation_meta(self, conversation_id: str) -> Dict[str, Any]:
        """
        For API responses: returns stage, requirements, missing, notes.
        """
        _, state = self._get_conversation(conversation_id)
        return {
            "stage": state.stage.name,
            "requirements": state.requirements,
            "missing": state.missing,
            "notes": state.notes,
        }

    # -------- Internal helpers --------

    def _get_conversation(self, conversation_id: str) -> Tuple[ChatHistory, ConversationState]:
        if conversation_id not in self._conversations:
            raise KeyError("Unknown conversation_id")
        return self._conversations[conversation_id]

    async def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_id] = lock
            return lock

    # ---------------------------
    # Stage handlers
    # ---------------------------
    async def _handle_collecting_requirements(
        self,
        history: ChatHistory,
        state: ConversationState,
        user_message: str,
    ) -> str:
        # 1) Quotation agent conversational response
        quote_reply = await self._invoke_agent_on_history(
            agent=self._quotation_agent,
            history=history,
            temperature=1,
        )
        history.add_assistant_message(quote_reply)

        # 2) Extract requirements update
        updated_req = await self._run_extractor(user_message=user_message, current=state.requirements)
        state.requirements = updated_req
        append_requirements_snapshot(history, state.requirements)

        # 3) Validate
        state.stage = ConversationStage.VALIDATING
        validation = await self._run_validator(history)
        state.last_validation = validation
        state.missing = validation.get("missing", []) or []
        state.notes = validation.get("notes", []) or []

        if validation.get("is_complete") is True:
            state.stage = ConversationStage.AWAITING_CONFIRMATION
            confirmation_msg = self._build_confirmation_prompt(state)
            history.add_assistant_message(confirmation_msg)
            return confirmation_msg

        state.stage = ConversationStage.COLLECTING_REQUIREMENTS
        return quote_reply

    async def _handle_validating(self, history: ChatHistory, state: ConversationState) -> str:
        validation = await self._run_validator(history)
        state.last_validation = validation
        state.missing = validation.get("missing", []) or []
        state.notes = validation.get("notes", []) or []

        if validation.get("is_complete") is True:
            state.stage = ConversationStage.AWAITING_CONFIRMATION
            msg = self._build_confirmation_prompt(state)
            history.add_assistant_message(msg)
            return msg

        state.stage = ConversationStage.COLLECTING_REQUIREMENTS
        msg = "I still need a bit more information:\n- " + "\n- ".join(state.missing or ["(unspecified)"])
        history.add_assistant_message(msg)
        return msg

    async def _handle_awaiting_confirmation(
        self,
        history: ChatHistory,
        state: ConversationState,
        user_message: str,
    ) -> str:
        if _looks_like_confirmation(user_message):
            state.stage = ConversationStage.GENERATING_FINAL_BOM
            return await self._handle_generating_final_bom(history, state)

        # user changed something
        state.stage = ConversationStage.COLLECTING_REQUIREMENTS
        return await self._handle_collecting_requirements(history, state, user_message)

    async def _handle_generating_final_bom(self, history: ChatHistory, state: ConversationState) -> str:
        generator_history = ChatHistory()
        generator_history.add_system_message(
            "You are a Cisco BOM generator. Only use valid Cisco SKUs when explicitly provided; otherwise use 'TBD'."
        )
        generator_history.add_user_message(
            json.dumps({"requirements": state.requirements}, ensure_ascii=False)
        )

        bom = await self._invoke_agent_on_history(
            agent=self._generator_agent,
            history=generator_history,
            temperature=1,
        )

        history.add_assistant_message(bom)
        state.stage = ConversationStage.COMPLETE
        return bom

    # ---------------------------
    # Agent invocation helpers
    # ---------------------------
    async def _invoke_agent_on_history(
        self,
        agent: ChatCompletionAgent,
        history: ChatHistory,
        *,
        temperature: float,
    ) -> str:
        execution_settings = self._kernel.get_prompt_execution_settings_from_service_id(self._service_id)
        if isinstance(execution_settings, OpenAIChatPromptExecutionSettings):
            execution_settings.temperature = temperature

        chat_service = self._kernel.get_service(service_id=self._service_id)
        response = await chat_service.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
        )
        return str(response)

    async def _run_extractor(self, user_message: str, current: RequirementsState) -> RequirementsState:
        extractor_history = ChatHistory()
        extractor_history.add_system_message("Return ONLY valid JSON. No prose. No markdown.")
        extractor_history.add_user_message(
            json.dumps(
                {
                    "current_requirements": current,
                    "user_latest_message": user_message,
                },
                ensure_ascii=False,
            )
        )

        out = await self._invoke_agent_on_history(self._extractor_agent, extractor_history, temperature=1)

        data = _safe_json_loads(out) or {}
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            items: List[RequirementItem] = []
            for it in data["items"]:
                if not isinstance(it, dict):
                    continue
                product_name = str(it.get("product_name", "")).strip()
                sku = it.get("sku", None)
                if sku is not None:
                    sku = str(sku).strip() or None
                try:
                    qty = int(it.get("quantity", 1))
                except Exception:
                    qty = 1
                if product_name and qty >= 1:
                    items.append({"product_name": product_name, "sku": sku, "quantity": qty})
            return {"items": items}

        return current

    async def _run_validator(self, history: ChatHistory) -> Dict[str, Any]:
        out = await self._invoke_agent_on_history(self._validation_agent, history, temperature=1)
        data = _safe_json_loads(out)
        if isinstance(data, dict):
            data.setdefault("is_complete", False)
            data.setdefault("missing", [])
            data.setdefault("notes", [])
            return data
        return {"is_complete": False, "missing": ["Validator returned invalid JSON"], "notes": []}

    def _build_confirmation_prompt(self, state: ConversationState) -> str:
        lines = ["I can now generate your BOM. Please confirm these items:"]
        for it in state.requirements.get("items", []):
            sku = it.get("sku") or "TBD"
            lines.append(f"- {it.get('quantity', 1)} × {it.get('product_name', '').strip()} (SKU: {sku})")
        if state.notes:
            lines.append("\nNotes:")
            lines.extend([f"- {n}" for n in state.notes])
        lines.append("\nReply 'yes' to generate the final BOM, or tell me what to change.")
        return "\n".join(lines)