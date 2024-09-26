import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Multi-Modal Decoder (Generates Text and Speech)
class MultiModalDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(MultiModalDecoder, self).__init__()
        self.hidden_size = hidden_size
        # Text Decoder
        self.text_decoder = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        # Speech Decoder (Placeholder)
        # In practice, use models like Tacotron2 or FastSpeech for speech synthesis
        self.speech_decoder = nn.Linear(hidden_size, hidden_size)  # Placeholder

    def forward(self, encoder_outputs):
        # Generate text output
        text_outputs = self.text_decoder(inputs_embeds=encoder_outputs)
        # Generate speech output (Placeholder)
        speech_outputs = self.speech_decoder(encoder_outputs[:, 0, :])  # Use first token
        return text_outputs, speech_outputs
