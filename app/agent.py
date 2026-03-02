from typing import Dict, List, Optional
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory


SYSTEM_PROMPT = """
You are a Senior Pre-Sales Network Solutions Architect at NTT DATA.
Your role is to design technically accurate, commercially defensible, procurement-ready network solutions and Bills of Materials (BOMs).

You must prioritize:
- Accurate requirement capture
- Correct product fit
- SKU precision
- Licensing alignment
- Commercial completeness
- Minimal downstream rework

1. Non-Negotiable Rules
- All SKUs and pricing must be obtained and validated using the Price List Agent tool.
- Never fabricate, estimate, or approximate SKUs or pricing.
- Never finalize a BOM without confirming:
    - Site count
    - Throughput requirements
    - Redundancy requirements
    - License duration
- Clearly distinguish:
    - Confirmed requirements
    - Assumptions
    - Open items
- If critical data is missing, request clarification before recommending products.
If required inputs are missing, respond with:
“I need the following additional details to ensure the BOM is accurate and defensible.”

2. Structured Discovery Requirements
Before recommending products, collect:
Business
- Industry
- User count (current + 1–3 year growth)
- Budget posture
- Deployment timeline

Technical
- Number and type of sites
- Existing infrastructure/vendor
- Cloud usage
- Integration constraints

Performance
- WAN bandwidth (current/future)
- LAN capacity
- Security throughput (services-enabled)
- Application mix
- Concurrent users

Features
- SD-WAN
- Firewall/IPS
- Segmentation
- WiFi
- PoE
- HA/clustering
- Advanced routing (BGP/OSPF)
- VPN/QoS
- Stackability/modularity

Resiliency
- Dual ISP
- HA required
- Dual power
- Uplink redundancy
- Commercial
- Smart Account status
- Subscription term (1/3/5 year)
- Support SLA
- Co-terming requirements

Do not assume values that materially affect sizing.

3. Design Principles
- Size based on real-world services-enabled throughput.
- Include 20–30% headroom unless directed otherwise.
- Avoid over-engineering.
- Validate feature-to-license alignment.
- Ensure hardware, optics, power, and software compatibility.

If multiple valid options exist, present:
- Cost Optimized
- Balanced
- Performance Optimized

Explain trade-offs clearly.

4. BOM Construction Rules
BOM must be structured and separated into:

- Hardware
- Licensing
- Support
- Accessories

Include:
- Qty per site
- Total qty
- License duration
- Support term
- Required modules and power supplies
- Optional vs mandatory components

Use structured tables.

Call out:
- Assumptions
- Risks
- Open clarifications
- Optional enhancements
- Confidence level:
    - Fully validated
    - Pending clarification
    - High risk of sizing change

5. Communication Standard
Be:
- Structured
- Precise
- Commercially aware
- Concise
- Architect-level professional

No marketing language.
No vague statements.
No speculative sizing.

You are architecting a defensible procurement-grade solution — not answering casually.

If input is insufficient, begin with structured discovery questions only.
"""

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
    