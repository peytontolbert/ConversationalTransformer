import torch
import torch.nn as nn
import torch.nn.functional as F

# Student-Teacher Framework for Knowledge Distillation
class StudentTeacherModel(nn.Module):
    def __init__(self, student_model, teacher_model, reward_model):
        super(StudentTeacherModel, self).__init__()
        self.student = student_model
        self.teacher = teacher_model  # Pretrained LLM as teacher
        self.reward_model = reward_model  # Instance of TransformerRewardModel
        self.criterion = nn.KLDivLoss(reduction='batchmean')  # Use KLDivLoss for knowledge distillation

    def forward(self, text_inputs, audio_tensor, video_input, unified_embedding, input_prompt):
        # Student forward pass
        student_outputs = self.student(unified_embedding, input_prompt)
        state_sequence, trajectory, text_outputs, speech_outputs, total_reward = student_outputs  # Unpack student outputs

        # Extract student logits
        student_logits = text_outputs.logits  # Ensure student model returns logits

        # Teacher forward pass (only on text for simplicity)
        with torch.no_grad():
            teacher_outputs = self.teacher(**text_inputs)
            teacher_logits = teacher_outputs.logits  # Extract teacher logits

        # Compute knowledge distillation loss
        loss = self.criterion(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1)
        )  # KLDivLoss between student and teacher probabilities

        # Compute reward based on cosine similarity
        reward = self.reward_model.get_reward(teacher_logits, student_logits)

        return student_outputs, loss, reward
