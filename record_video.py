import cv2
import time
import pyautogui
import numpy as np
import torch
from transformers import ViTImageProcessor, GPT2LMHeadModel, VideoMAEModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Video
video_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# Device configuration
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device_type)
# Define a consistent hidden size for all modality models
# This should match the hidden size used in your StudentTeacherModel and OmniModalTransformer
hidden_size = text_model.config.hidden_size


video_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device_type)

def record_screen(filename="user_screen.mp4", duration=8, desired_num_frames=160):
    print(f"Recording screen for {duration} seconds.")
    # OpenCV screen recording setup
    screen_width = 1920
    screen_height = 1080
    fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (screen_width, screen_height))

    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            img = pyautogui.screenshot()
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
    except Exception as e:
        logger.error(f"Failed to record screen: {e}")  # {{ edit_5 }}
        # Create dummy video embeddings in case of failure
        video_embeddings = torch.zeros((1, hidden_size), dtype=torch.float).to(device_type)
        return video_embeddings
    
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
        # Process frame with video_feature_extractor
        processed_frame = video_feature_extractor(images=frame, return_tensors="pt")['pixel_values']
        frames.append(processed_frame)
    cap.release()

    print(f"Total frames captured: {len(frames)}")  # {{ edit_1 }}

    # Resample frames to desired_num_frames
    if len(frames) > desired_num_frames:
        indices = np.linspace(0, len(frames)-1, desired_num_frames).astype(int)
        frames = [frames[i] for i in indices]
        print(f"Resampled frames to {desired_num_frames}")  # {{ edit_2 }}
    elif len(frames) < desired_num_frames:
        if len(frames) == 0:
            # If no frames were captured, create dummy frames
            dummy_frame = torch.zeros((1, 3, 224, 224), dtype=torch.float).to(device_type)
            frames = [dummy_frame for _ in range(desired_num_frames)]
            print("No frames captured; filled with zeros.")  # {{ edit_3 }}
        else:
            # Pad with the last captured frame to reach desired_num_frames
            last_frame = frames[-1]
            while len(frames) < desired_num_frames:
                frames.append(last_frame)
            print(f"Padded frames to {desired_num_frames}")  # {{ edit_4 }}

    # Permute to match VideoMAEModel's expected input shape
    video_input = torch.cat(frames, dim=0).permute(0, 2, 3, 1).unsqueeze(0).to(device_type)  # Changed permutation to [Batch, Frames, Channels, Height, Width]
    print(f"video_input shape after to(device): {video_input.shape}")  # Should have shape [1, Frames, Channels, Height, Width]

    # Process video through video encoder
    try:
        with torch.no_grad():
            video_embeddings = video_encoder(video_input).last_hidden_state  # Expected shape: [1, num_tokens, hidden_size]
        print(f"video_embeddings shape: {video_embeddings.shape}")  # Added for debugging
        
        # Optionally aggregate video embeddings (e.g., mean pooling)
        video_embeddings = video_embeddings.mean(dim=1)  # Shape: [1, hidden_size]
        print(f"Aggregated video_embeddings shape: {video_embeddings.shape}")  # Added for debugging
    except Exception as e:
        logger.error(f"Failed to process video through video_encoder: {e}")  # {{ edit_5 }}
        # Create dummy video embeddings in case of failure
        video_embeddings = torch.zeros((1, hidden_size), dtype=torch.float).to(device_type)
        print("Using dummy video embeddings due to video_encoder processing failure.")  # {{ edit_6 }}

    return video_embeddings  # Return video embeddings
