import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    ViTImageProcessor,
    Wav2Vec2Processor,
    GPT2Tokenizer,
    GPT2Model,
    AdamW,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import requests
import threading
import time
import cv2
import pyaudio
import wave
import logging
import faiss
import torchaudio
import pyautogui
import whisper  # {{ Added for Whisper integration }}
from transformers import VideoMAEModel, VideoMAEConfig  # {{ Added for Video Encoder }}

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from studentteacher import StudentTeacherModel
from omnitransformer import OmniModalTransformer

# Import additional modules for unsupervised learning
import torch.optim as optim
from torch.nn import MSELoss

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
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_model = AutoModel.from_pretrained("bert-base-uncased").to(device_type)
logger.info("Loaded text tokenizer and model.")

# Audio
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device_type)
logger.info("Loaded audio processor and model.")

# Video
video_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
video_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device_type)
logger.info("Loaded video image processor and model.")

# Import and initialize the pretrained LLM
# Initialize GPT-2 tokenizer and model for target generation
llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
llm_model = GPT2Model.from_pretrained("gpt2").to(device_type)
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
teacher_model = AutoModel.from_pretrained("bert-large-uncased").to(device_type)
model = StudentTeacherModel(student_model, teacher_model).to(device_type)
logger.info("Initialized Student-Teacher Model.")

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# Initialize unsupervised learning components
# Define unsupervised loss function
unsup_loss_function = MSELoss()

# Initialize Whisper model
whisper_model = whisper.load_model("base")  # You can choose appropriate model size

# Initialize Video Encoder
video_encoder_config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
video_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device_type)
logger.info("Loaded Video Encoder model.")

import threading  # {{ Added for threading }}

# Update DURATION to capture 160 frames at 20 FPS
DURATION = 8  # Increased from 5 to 8 seconds

def record_audio_and_screen(audio_filename, screen_filename, duration):
    audio_result = {}
    screen_result = {}

    def audio_thread():
        audio_tensor, audio_text_inputs = record_audio(audio_filename, duration)
        audio_result['audio_tensor'] = audio_tensor
        audio_result['audio_text_inputs'] = audio_text_inputs

    def screen_thread():
        video_embeddings = record_screen(screen_filename, duration)
        screen_result['video_embeddings'] = video_embeddings

    # Start both threads
    thread_audio = threading.Thread(target=audio_thread)
    thread_screen = threading.Thread(target=screen_thread)

    thread_audio.start()
    thread_screen.start()

    # Wait for both threads to finish
    thread_audio.join()
    thread_screen.join()

    return audio_result['audio_tensor'], audio_result['audio_text_inputs'], screen_result['video_embeddings']

def record_audio(filename="user_audio.wav", duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    # List all available input devices
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        device = p.get_device_info_by_host_api_device_index(0, i)
        if device.get('maxInputChannels') > 0:
            print(f"Input Device id {i} - {device.get('name')}")

    # Select the Logi Webcam Microphone (replace with your device ID if different)
    audio_index = 1  # Change this to 2 to select the Logi C270 HD Webcam microphone
    print(f"Using input device id {audio_index}")

    logger.info(f"Recording audio for {duration} seconds.")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=audio_index)  # Specify device index

    frames = []
    start_time = time.time()

    while time.time() - start_time < duration:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Load audio using torchaudio
    try:
        print(f"Loading audio file with torchaudio: {filename}")  # {{ edit_7 }}
        waveform, sample_rate = torchaudio.load(filename)  # {{ edit_1 }}
        if sample_rate != RATE:
            # Resample if necessary
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=RATE)
            waveform = resampler(waveform)
            print(f"Resampled waveform shape: {waveform.shape}")  # {{ edit_resample }}
        # If stereo, convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print(f"Converted to mono waveform shape: {waveform.shape}")  # {{ edit_mono }}
        audio_tensor = waveform.to(device_type)  # {{ edit_2 }}
        print(f"audio_tensor shape: {audio_tensor.shape}")  # {{ edit_1 }}

        transcript = whisper_model.transcribe(filename)
        print(f"Transcription result: {transcript}")  # {{ edit_3 }}
        audio_text = transcript['text']
        print(f"Transcribed text: {audio_text}")  # {{ edit_4 }}
        
        # Tokenize transcribed text for the teacher model
        audio_text_inputs = text_tokenizer(
            audio_text, return_tensors='pt', truncation=True, max_length=128
        ).to(device_type)
    except Exception as e:
        logger.error(f"Failed to load or transcribe audio with torchaudio: {e}")  # {{ edit_5 }}
        audio_tensor = torch.zeros((1, RATE * duration), dtype=torch.float).to(device_type)  # {{ edit_6 }}
        audio_text_inputs = None  # Handle absence of transcription
    
    return audio_tensor, audio_text_inputs  # Return tensors only

def record_screen(filename="user_screen.mp4", duration=5):
    print(f"Recording screen for {duration} seconds.")
    # OpenCV screen recording setup
    screen_width = 1920
    screen_height = 1080
    fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (screen_width, screen_height))

    start_time = time.time()

    while time.time() - start_time < duration:
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)

    out.release()

    # Load video frames
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = video_feature_extractor(images=frame, return_tensors="pt")['pixel_values']
        frames.append(frame)
    cap.release()
    video_input = torch.cat(frames, dim=0).unsqueeze(0).to(device_type)
    print(f"video_input shape after to(device): {video_input.shape}")  # Should now have 5 dimensions
    
    # Process video through video encoder
    with torch.no_grad():
        video_embeddings = video_encoder(video_input).last_hidden_state  # Shape: [num_frames, hidden_size]
    print(f"video_embeddings shape: {video_embeddings.shape}")  # Added for debugging
    
    # Optionally aggregate video embeddings (e.g., mean pooling)
    video_embeddings = video_embeddings.mean(dim=0, keepdim=True)  # Shape: [1, hidden_size]
    print(f"Aggregated video_embeddings shape: {video_embeddings.shape}")  # Added for debugging
    
    return video_embeddings  # Return video embeddings

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
            audio_tensor, audio_text_inputs, video_embeddings = record_audio_and_screen("user_audio.wav", "user_screen.mp4", DURATION)
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
        text_embeddings = text_model(**text_inputs).last_hidden_state  # (1, seq_len, hidden_size)
        # Audio embeddings
        audio_inputs = audio_processor(audio_tensor.squeeze(0), sampling_rate=16000, return_tensors="pt")
        audio_inputs = {k: v.to(device_type) for k, v in audio_inputs.items()}
        audio_embeddings = audio_model(**audio_inputs).last_hidden_state  # (1, seq_len, hidden_size)
        # Video embeddings are already obtained
        # Transcribed audio text embeddings
        if audio_text_inputs is not None:
            audio_text_embeddings = text_model(**audio_text_inputs).last_hidden_state  # (1, seq_len, hidden_size)
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
            raise ValueError(f"Hidden size mismatch among embeddings: {hidden_sizes}")

        # Example projection layers if necessary
        projection_layer_audio = nn.Linear(audio_embeddings.size(-1), hidden_size).to(device_type)
        projection_layer_video = nn.Linear(video_embeddings.size(-1), hidden_size).to(device_type)
        projection_layer_audio_text = nn.Linear(audio_text_embeddings.size(-1), hidden_size).to(device_type)

        audio_embeddings = projection_layer_audio(audio_embeddings)
        video_embeddings = projection_layer_video(video_embeddings)
        audio_text_embeddings = projection_layer_audio_text(audio_text_embeddings)

        # Now concatenate
        unified_embedding = torch.cat(
            [text_embeddings, audio_embeddings, video_embeddings, audio_text_embeddings], dim=1  # Concatenate along sequence length
        )  # Shape: (batch_size, total_seq_len, hidden_size)

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
        _, loss = model(text_inputs, audio_tensor, video_input, unified_embedding)

        total_loss = loss 

        print(f"Total Loss: {total_loss.item()}")

        optimizer.zero_grad()
        total_loss.backward()

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
