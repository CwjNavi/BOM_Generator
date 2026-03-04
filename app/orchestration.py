from enum import Enum

class ConversationStage(Enum):
    COLLECTING_REQUIREMENTS = 1
    VALIDATING = 2
    AWAITING_CONFIRMATION = 3
    GENERATING_FINAL_BOM = 4
    COMPLETE = 5


class QuotationOrchestrator:

    def __init__(self, quotation_agent, validation_agent, bom_generator_agent):
        self.stage = ConversationStage.COLLECTING_REQUIREMENTS
        self.quotation_agent = quotation_agent
        self.validation_agent = validation_agent
        self.bom_generator_agent = bom_generator_agent
        self.preliminary_bom = None

    async def handle_message(self, user_input, chat_history, execution_settings):

        chat_history.add_user_message(user_input)

        # ✅ Stage 1 — Collect requirements → Generate Preliminary BOM
        if self.stage == ConversationStage.COLLECTING_REQUIREMENTS:
            response = await self.quotation_agent.get_response(
                chat_history=chat_history,
                execution_settings=execution_settings
            )

            self.preliminary_bom = response.content
            chat_history.add_assistant_message(self.preliminary_bom)

            self.stage = ConversationStage.VALIDATING
            return self.preliminary_bom


        # ✅ Stage 2 — Validate BOM vs requirements
        elif self.stage == ConversationStage.VALIDATING:
            response = await self.validation_agent.get_response(
                chat_history=chat_history,
                execution_settings=execution_settings
            )

            chat_history.add_assistant_message(response.content)

            if "Would you like to generate the final BOM" in response.content:
                self.stage = ConversationStage.AWAITING_CONFIRMATION

            return response.content


        # ✅ Stage 3 — Wait for confirmation
        elif self.stage == ConversationStage.AWAITING_CONFIRMATION:

            if "generate" in user_input.lower():
                self.stage = ConversationStage.GENERATING_FINAL_BOM

                response = await self.bom_generator_agent.get_response(
                    chat_history=chat_history,
                    execution_settings=execution_settings
                )

                chat_history.add_assistant_message(response.content)
                self.stage = ConversationStage.COMPLETE
                return response.content

            else:
                return "Please confirm if you would like to generate the final BOM."


        return "Conversation complete."