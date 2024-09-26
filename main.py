import torch
import torch.nn as nn
import whisper
from transformers import (
    AutoModel,
    ViTImageProcessor,
    Wav2Vec2Processor,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
)
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
# Import additional modules for unsupervised learning
import numpy as np
import logging
import torchaudio
import warnings
from studentteacher import StudentTeacherModel
from omnitransformer import OmniModalTransformer
from transformers import VideoMAEModel, VideoMAEConfig  # {{ Added for Video Encoder }}
from reward import TransformerRewardModel 
from record_audio_and_screen import record_audio_and_screen
from record_video import record_screen  # Ensure you import necessary functions
# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# List available torchaudio backends
logger.info("Available torchaudio backends: %s", torchaudio.list_audio_backends())

# Device configuration
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device_type}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define modality-specific tokenizers and models
# Text
text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text_model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device_type)
text_model.train()  # Set student model to train mode
logger.info("Loaded aligned text tokenizer and model.")

# Audio
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = AutoModel.from_pretrained(
    "facebook/wav2vec2-base-960h",
    output_hidden_states=True  # {{ edit_1 }}
).to(device_type)
logger.info("Loaded audio processor and model.")

# Video
video_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
video_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device_type)
logger.info("Loaded video image processor and model.")

# Import and initialize the pretrained LLM
# Initialize GPT-2 tokenizer and model for target generation
llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
llm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device_type)
llm_model.eval()  # Set LLM to evaluation mode since it's used for target generation
logger.info("Loaded pretrained LLM for target generation.")

# Define llm_hidden_size based on the LLM's configuration
llm_hidden_size = llm_model.config.hidden_size
logger.info(f"LLM Hidden Size: {llm_hidden_size}")

# Define a consistent hidden size for all modality models
# This should match the hidden size used in your StudentTeacherModel and OmniModalTransformer
hidden_size = text_model.config.hidden_size
logger.info(f"Using hidden size: {hidden_size}")

# Pass llm_hidden_size to OmniModalTransformer
api_url = ""  # Make sure to provide the actual API URL
student_model = OmniModalTransformer(hidden_size, llm_hidden_size, api_url).to(device_type)
teacher_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device_type)
teacher_model.eval()  # Set teacher model to evaluation mode

# Initialize Reward Model
reward_model = TransformerRewardModel(model_name="gpt2")  # Initialize with appropriate model
# Initialize Student-Teacher Model with Reward Model
model = StudentTeacherModel(student_model, teacher_model, reward_model).to(device_type)
logger.info("Initialized Student-Teacher Model with Reward Model.")

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# Initialize unsupervised learning components
# Define unsupervised loss function
unsup_loss_function = MSELoss()

# Initialize Whisper model
whisper_model = whisper.load_model("base")  # You can choose appropriate model size

# Centralized Video Encoder Initialization
video_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device_type)
logger.info("Loaded Video Encoder model.")

# Update DURATION to capture 160 frames at 20 FPS
DURATION = 8  # Increased from 5 to 8 seconds
desired_num_frames = 160  # Set desired_num_frames to a multiple of batch_size (16)

# Real-Time Interaction Loop
def real_time_interaction():
    print("Assistant is ready. Type 'exit' to quit.")
    model.train()  # Set model to training mode
    while True:
        user_choice = input("Enter 'r' to record audio and screen, 't' to input text, or 'exit' to quit: ")
        if user_choice.lower() == 'exit':
            break
        elif user_choice.lower() == 'r':
            # Record audio and screen in parallel
            audio_tensor, audio_text_inputs, video_embeddings = record_audio_and_screen(
                "user_audio.wav", "user_screen.mp4", DURATION, desired_num_frames=desired_num_frames
            )
            video_input = video_embeddings  # Define video_input
            
            # Optional text input
            text_input_str = input("Enter additional text (or press Enter to skip): ")
            if text_input_str.strip() == '':
                text_inputs = None
            else:
                text_inputs = text_tokenizer(
                    text_input_str, return_tensors='pt', truncation=True, max_length=128
                ).to(device_type)
        elif user_choice.lower() == 't':
            # Text input only
            text_input_str = input("Enter your text: ")
            text_inputs = text_tokenizer(
                text_input_str, return_tensors='pt', truncation=True, max_length=128
            ).to(device_type)
            audio_tensor = None
            audio_text_inputs = None
            video_input = None  # Ensure video_input is defined
        else:
            print("Invalid choice. Please try again.")
            continue

        # Handle missing inputs
        if text_inputs is None:
            # Create dummy text inputs
            text_inputs = {
                'input_ids': torch.zeros((1, 1), dtype=torch.long).to(device_type),
                'attention_mask': torch.ones((1, 1), dtype=torch.long).to(device_type)
            }
        if audio_tensor is None:
            # Create dummy audio inputs
            audio_tensor = torch.zeros((1, 16000), dtype=torch.float).to(device_type)
            audio_text_inputs = None
        if audio_text_inputs is None and user_choice.lower() == 'r':
            # Create dummy audio text inputs
            audio_text_inputs = {
                'input_ids': torch.zeros((1, 1), dtype=torch.long).to(device_type),
                'attention_mask': torch.ones((1, 1), dtype=torch.long).to(device_type)
            }
        if video_input is None:
            # Create dummy video embeddings
            video_input = torch.zeros((1, hidden_size), dtype=torch.float).to(device_type)  # Ensure video_input is a Tensor

        # Preprocess inputs
        # Text embeddings
        text_embeddings = text_model(**text_inputs).hidden_states[-1]  # (1, seq_len, hidden_size)
        # Audio embeddings
        audio_inputs = audio_processor(audio_tensor.squeeze(0), sampling_rate=16000, return_tensors="pt")
        audio_inputs = {k: v.to(device_type) for k, v in audio_inputs.items()}
        audio_output = audio_model(**audio_inputs)
        if audio_output.hidden_states is not None:
            audio_embeddings = audio_output.hidden_states[-1]  # (1, seq_len, hidden_size)
        else:
            logger.error("audio_model did not return hidden_states.")
            # Handle the absence of hidden_states appropriately
            audio_embeddings = torch.zeros((1, 1, hidden_size), dtype=torch.float).to(device_type)
        # Video embeddings are already obtained
        # Transcribed audio text embeddings
        if audio_text_inputs is not None:
            audio_text_embeddings = text_model(**audio_text_inputs).hidden_states[-1]  # (1, seq_len, hidden_size)
            print(f"audio_text_embeddings shape: {audio_text_embeddings.shape}")  # Added for debugging
        else:
            audio_text_embeddings = torch.zeros((1, 1, hidden_size), dtype=torch.float).to(device_type)

        # Debug: Print tensor shapes before concatenation
        print(f"text_embeddings shape: {text_embeddings.shape}")
        print(f"audio_embeddings shape: {audio_embeddings.shape}")
        print(f"video_embeddings shape: {video_embeddings.shape}")
        print(f"audio_text_embeddings shape: {audio_text_embeddings.shape}")  # Added for debugging

        # Ensure all embeddings have the same batch size
        batch_sizes = [emb.shape[0] for emb in [text_embeddings, audio_embeddings, video_embeddings, audio_text_embeddings]]
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"Batch size mismatch among embeddings: {batch_sizes}")

        # Ensure all embeddings have the same hidden size
        hidden_sizes = [emb.shape[2] if emb.ndim == 3 else emb.shape[1] for emb in [text_embeddings, audio_embeddings, video_embeddings, audio_text_embeddings]]
        if len(set(hidden_sizes)) != 1:
            logger.error(f"Hidden size mismatch among embeddings: {hidden_sizes}")
            raise ValueError(f"Hidden size mismatch among embeddings: {hidden_sizes}")
        
        # Example projection layers if necessary
        projection_layer_audio = nn.Linear(audio_embeddings.size(-1), hidden_size).to(device_type)
        projection_layer_video = nn.Linear(video_embeddings.size(-1), hidden_size).to(device_type)
        projection_layer_audio_text = nn.Linear(audio_text_embeddings.size(-1), hidden_size).to(device_type)

        audio_embeddings = projection_layer_audio(audio_embeddings)
        video_embeddings = projection_layer_video(video_embeddings)
        audio_text_embeddings = projection_layer_audio_text(audio_text_embeddings)

        # Ensure video_embeddings has 3 dimensions
        video_embeddings = video_embeddings.unsqueeze(1)  # Shape: [1, 1, hidden_size]
        print(f"video_embeddings shape after unsqueeze: {video_embeddings.shape}")  # Debug

        unified_embedding = torch.cat(
            [text_embeddings, audio_embeddings, video_embeddings, audio_text_embeddings], dim=1
        )
        print(f"unified_embedding shape: {unified_embedding.shape}")  # Debug

        # Generate response
        input_prompt = text_input_str if (user_choice.lower() == 't' and text_input_str.strip() != '') else "User provided audio and video inputs."
        try:
            state_sequence, trajectory, text_outputs, speech_outputs, total_reward = student_model(
                unified_embedding, input_prompt
            )
            print(f"Assistant: {state_sequence[-1]}")
            print(f"Total Reward: {total_reward}")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            continue

        # Decode text output
        generated_text = text_tokenizer.decode(
            text_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True
        )
        print(f"Assistant (Text Output): {generated_text}")

        # Handle speech output (Placeholder)
        print("Assistant (Speech Output): [Speech output generated]")

        # Real-time learning (update model parameters)
        student_outputs, loss, reward = model(
            text_inputs,
            audio_tensor,
            video_input,
            unified_embedding,
            input_prompt  # {{ edit: pass input_prompt to the model }}
        )

        total_loss = loss 
        
        print(f"Total Loss: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # {{ edit_7 }}

        # Optional: If using a separate optimizer for unsupervised components
        # unsup_optimizer.zero_grad()
        # unsup_loss.backward()
        # unsup_optimizer.step()

        # Debug: Check gradients before optimizer step
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm().item()}")
            else:
                print(f"Gradient for {name} is None.")

        optimizer.step()
        scheduler.step()

        # Update Memory System
        student_model.memory_system.add_memory(unified_embedding.mean(dim=1), input_prompt)

# Example usage
if __name__ == "__main__":

    # Run the assistant
    real_time_interaction()
