import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy Network (Determines Actions)
class TransformerPolicyNetwork(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(TransformerPolicyNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.transformer = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        return logits

    def generate_thought(self, input_prompt, max_length=50):  # {{ edit_1 }}
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        logits = self.forward(input_ids, attention_mask)
        # Sample thought from logits
        probs = F.softmax(logits[:, -1, :], dim=-1)
        action_id = torch.multinomial(probs, num_samples=1)
        action_text = self.tokenizer.decode(action_id[0])
        student_logits = logits  # {{ edit_2 }}
        return action_text, student_logits, input_ids, attention_mask
