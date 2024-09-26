import torch.nn as nn
from tool import ToolInteractionModule
from memory import MemorySystem
from policy import TransformerPolicyNetwork
from reward import TransformerRewardModel
from iterationofthought import IterationOfThought
from decoder import MultiModalDecoder
from encoder import LiquidTransformerEncoder
from transformers import GPT2LMHeadModel  # Ensure necessary imports if required

# Omni-Modal Transformer Model
class OmniModalTransformer(nn.Module):
    def __init__(self, hidden_size, llm_hidden_size, api_url):
        super(OmniModalTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.liquid_encoder = LiquidTransformerEncoder(hidden_size)
        self.multi_modal_decoder = MultiModalDecoder(hidden_size)
        self.tool_module = ToolInteractionModule(api_url)
        self.memory_system = MemorySystem(embedding_dim=hidden_size)
        # Policy and Reward Models
        self.policy_network = TransformerPolicyNetwork(hidden_size, vocab_size=50257)
        self.reward_model = TransformerRewardModel()
        self.iteration_of_thought = IterationOfThought(
            policy_network=self.policy_network,
            reward_model=self.reward_model
        )
        
        # Add projection layer to align with LLM's hidden size
        self.projection = nn.Linear(hidden_size, llm_hidden_size)

    def forward(self, embeddings, input_prompt):
        # Pass through the liquid transformer encoder
        encoder_outputs = self.liquid_encoder(embeddings)
        # Apply projection if necessary
        encoder_outputs = self.projection(encoder_outputs)
        # Run Iteration of Thought
        state_sequence, trajectory, total_reward = self.iteration_of_thought.run(input_prompt)
        # Decode outputs
        text_outputs, speech_outputs = self.multi_modal_decoder(encoder_outputs)
        return state_sequence, trajectory, text_outputs, speech_outputs, total_reward

    def interact_with_tool(self, tool_name, params):
        return self.tool_module.call_tool(tool_name, params)
