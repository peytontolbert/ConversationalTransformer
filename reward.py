import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reward Model (Pretrained, Not Trained Further)
class TransformerRewardModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def get_reward(self, state_action_text):
        # Use prompting to get a reward score
        input_ids = self.tokenizer.encode(state_action_text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        # Convert loss to reward (lower loss implies better output)
        reward = -loss
        return reward
