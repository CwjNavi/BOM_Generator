from __future__ import annotations

import json
from typing import List, Optional, TypedDict

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents.chat_history import ChatHistory

REQUIREMENTS_MARKER = "##CURRENT_REQUIREMENTS_JSON##"


class RequirementItem(TypedDict, total=False):
    product_name: str
    sku: Optional[str]  # None if unknown / not confirmed
    quantity: int


class RequirementsState(TypedDict):
    items: List[RequirementItem]


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None


def get_latest_requirements(history: ChatHistory) -> RequirementsState:
    """Scan ChatHistory backwards for the latest requirements snapshot."""
    for msg in reversed(history.messages):
        if msg.role.lower() == "system" and msg.content and REQUIREMENTS_MARKER in msg.content:
            parts = msg.content.split(REQUIREMENTS_MARKER, 1)
            payload = parts[1].strip()
            data = _safe_json_loads(payload)
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                return {"items": data["items"]}
    return {"items": []}


def append_requirements_snapshot(history: ChatHistory, requirements: RequirementsState) -> None:
    """Append a new snapshot into history (system message)."""
    history.add_system_message(
        f"{REQUIREMENTS_MARKER}\n{json.dumps(requirements, ensure_ascii=False, indent=2)}"
    )


# ---------------------------
# Agents
# ---------------------------

quotation_agent = ChatCompletionAgent(
    name="CiscoQuotationAgent",
    description="Gathers Cisco networking requirements via conversation.",
    instructions="""
You are an AI-powered quote assistant for Cisco networking hardware. Your job is to help customers create accurate Bills of Materials (BOM).

CORE RESPONSIBILITIES:
1. Gather requirements through natural conversation
2. Identify product needs (switches, routers, access points, etc.)
3. Extract key specifications:
   - Product Line (Switching, Wireless, Routing, Security)
   - Model preferences (if any)
   - Quantity needed
   - Port count requirements
   - PoE requirements (PoE+, PoE++, UPoE)
   - DNA License tier (Essentials, Advantage, Premier)
   - Support contract preferences
   - Power supply requirements

STATE MANAGEMENT (IMPORTANT):
- The system maintains a hidden "current requirements" JSON snapshot inside the chat history.
- You do NOT print JSON to the customer.
- Your job is to ask the right clarifying questions and confirm assumptions.

IMPORTANT RULES:
- Be conversational and helpful
- Only ask for information relevant to Cisco products
- Don't make up SKUs - only use valid Cisco part numbers
- If you're unsure about a SKU, ask for clarification
- When you have enough info, say "I can now generate your BOM" and list what you'll include
""".strip(),
)

extractor_agent = ChatCompletionAgent(
    name="CiscoRequirementsExtractor",
    description="Extracts/updates structured Cisco requirements JSON from the user's message.",
    instructions="""
You extract Cisco product requirements from the user's latest message and update the existing requirements JSON.

You will be given:
1) The latest requirements snapshot JSON (may be empty)
2) The user's latest message

OUTPUT RULES (VERY IMPORTANT):
- Output ONLY valid JSON. No prose, no markdown, no code fences.
- Output schema:
  {
    "items": [
      {
        "product_name": "<string>",
        "sku": "<string or null>",
        "quantity": <int>
      }
    ]
  }

MERGE RULES:
- If the user adds a new item, append it.
- If the user changes quantity for an existing product/SKU, update it.
- If you can confidently match an item by SKU, treat it as the same item.
- If SKU is not explicitly provided/confirmed, set sku to null. Do NOT invent SKUs.
- Quantity must be an integer >= 1. If user is vague, infer best integer (e.g., "a couple"=2), otherwise leave unchanged.
- Keep the list concise: only include product_name/sku/quantity.
- You may normalize product_name (e.g., "Catalyst 9300" instead of full sentence).
""".strip(),
)

validation_agent = ChatCompletionAgent(
    name="CiscoRequirementsValidator",
    description="Validates if requirements are sufficient to generate a Cisco BOM and flags missing info.",
    instructions=f"""
You validate whether the gathered Cisco requirements are sufficient to generate a BOM.

You will be given:
- The FULL chat history (including the hidden requirements snapshot marker {REQUIREMENTS_MARKER})

Your job:
1) Find the latest requirements JSON snapshot in the history.
2) Decide if enough info is present to produce a reasonable BOM.
3) Output ONLY valid JSON (no prose).

OUTPUT SCHEMA:
{{
  "is_complete": <true|false>,
  "missing": [ "<string>", ... ],
  "notes": [ "<string>", ... ]
}}

VALIDATION GUIDELINES:
- At minimum, to generate a BOM you need at least one item with:
  - product_name
  - quantity
- SKU can be null if the customer has not selected a specific SKU yet; note it as a potential blocker.
- Check if critical specs are missing for the product types mentioned:
  - Switching: port counts, PoE, uplinks, power supplies, DNA license tier, support contract
  - Wireless: AP model class, quantity, controller/cloud preference, PoE, support
  - Routing/Security: throughput, WAN interfaces, licensing/support
- If the user asked for "quote/BOM now" but missing key specs, set is_complete=false and list concise missing questions.
- If requirements look sufficient, set is_complete=true and missing=[].

Do NOT invent SKUs or products.
""".strip(),
)

generator_agent = ChatCompletionAgent(
    name="CiscoBOMGenerator",
    description="Formats the final requirements snapshot into a BOM table.",
    instructions=f"""
You generate a BOM table from the latest requirements JSON snapshot stored in chat history.

You will be given:
- The FULL chat history (including the hidden requirements snapshot marker {REQUIREMENTS_MARKER})

Steps:
1) Locate the latest requirements JSON snapshot.
2) Produce a markdown table with columns:
   - Product
   - SKU
   - Quantity
3) If sku is null, display 'TBD' in the SKU column.

OUTPUT RULES:
- Output ONLY markdown (no surrounding prose).
- Do NOT invent SKUs. Use 'TBD' when unknown.
""".strip(),
)