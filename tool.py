import requests
# Tool Interaction Module with External API Interaction
class ToolInteractionModule:
    def __init__(self, api_url):
        self.api_url = api_url

    def call_tool(self, tool_name, params):
        # Interact with the external API
        payload = {
            'tool_name': tool_name,
            'params': params
        }
        response = requests.post(self.api_url, json=payload)
        return response.json()
