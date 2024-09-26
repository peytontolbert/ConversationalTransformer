import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import logging  # Added import for logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Reward Model Using Cosine Similarity
class TransformerRewardModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def get_teacher_logits(self, state):  # {{ edit_1 }}
        input_ids = self.tokenizer.encode(state, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            teacher_logits = outputs.logits
        # Decode logits to text and print
        decoded_output = self.tokenizer.decode(torch.argmax(teacher_logits, dim=-1)[0], skip_special_tokens=True)
        logger.info(f"Decoded teacher output: {decoded_output}")
        return teacher_logits

    def get_reward(self, state, student_logits):  # {{ edit_2 }}
        """
        Compute reward based on cosine similarity between teacher and student logits.

        Args:
            state (str): The current state or prompt.
            student_logits (torch.Tensor): Logits from the student model.

        Returns:
            float: Reward score.
        """
        teacher_logits = self.get_teacher_logits(state)

        # Debug: Log the original sequence lengths
        student_seq_len = student_logits.size(1)
        teacher_seq_len = teacher_logits.size(1)
        logger.info(f"Student logits sequence length: {student_seq_len}")
        logger.info(f"Teacher logits sequence length: {teacher_seq_len}")

        # Align sequence lengths by truncating to the minimum sequence length
        min_seq_len = min(student_seq_len, teacher_seq_len)
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]

        # Debug: Log the truncated sequence length
        logger.info(f"Truncated sequence length to: {min_seq_len}")

        # Normalize logits
        teacher_norm = F.normalize(teacher_logits, p=2, dim=-1)
        student_norm = F.normalize(student_logits, p=2, dim=-1)

        # Compute cosine similarity
        similarity = cosine_similarity(student_norm, teacher_norm, dim=-1)

        # Average similarity over the sequence
        average_similarity = similarity.mean().item()

        # Scale reward to a suitable range
        reward = average_similarity  # This will be between -1 and 1

        logger.info(f"Computed reward: {reward}")

        return reward
