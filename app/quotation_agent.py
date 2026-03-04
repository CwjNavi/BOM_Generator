quotation_agent = ChatCompletionAgent(
    kernel=kernel,
    name="CiscoQuotationAgent",
    instructions="""
You are a Cisco Solutions Architect.

Based on the client's requirements, generate a PRELIMINARY BOM.

Return ONLY valid JSON in this format:

{
  "customer_requirements_summary": "...",
  "products": [
    {
      "product_name": "...",
      "sku": "...",
      "reason": "why this product was selected"
    }
  ],
  "assumptions": []
}

Rules:
- Do not include markdown
- Do not explain outside JSON
- If requirements are missing, list assumptions
"""
)

# chat_history = ChatHistory()
# chat_history.add_user_message(
#     "Customer needs 2 branch routers supporting SD-WAN and 48-port PoE switches for 100 users."
# )

# response = await quotation_agent.get_response(
#     chat_history=chat_history,
#     execution_settings=execution_settings
# )

# preliminary_bom_json = response.content
# print(preliminary_bom_json)