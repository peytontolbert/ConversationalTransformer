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

# **Added: Check expected number of channels from VideoMAEModel configuration**
expected_channels = getattr(video_encoder.config, 'num_channels', 3)  # Default to 3 if not present
print(f"VideoMAEModel expects {expected_channels} channels.")

def record_screen(filename="user_screen.mp4", duration=8, desired_num_frames=160):
    print(f"Recording screen for {duration} seconds.")
    # Dynamic screen resolution
    screen_width, screen_height = pyautogui.size()
    logger.info(f"Screen resolution: {screen_width}x{screen_height}")
    desired_fps = desired_num_frames / duration
    logger.info(f"Recording at {desired_fps} FPS.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, desired_fps, (screen_width, screen_height))

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
        processed_frame = processed_frame.float()  # Ensure float
        processed_frame = processed_frame.to(device_type)  # Ensure on correct device

        # **Added: Ensure processed_frame is a dense tensor**
        if processed_frame.is_sparse:
            processed_frame = processed_frame.to_dense()
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

    # **Ensuring all frames are dense and on the correct device**
    for i, frame in enumerate(frames):
        if frame.is_sparse:
            frames[i] = frame.to_dense()

    # Batch processing of frames
    batch_size = 16  # {{ edit_1: Define batch size }}
    num_batches = desired_num_frames // batch_size  # {{ edit_2: Calculate number of batches }}
    video_embeddings_list = []  # {{ edit_3: Initialize list to store embeddings }}

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_frames = frames[start_idx:end_idx]  # {{ edit_4: Slice frames for the current batch }}
        
        frames_tensor = torch.cat(batch_frames, dim=0)  # Shape: [16, 3, 224, 224]
        frames_tensor = frames_tensor.to(device_type)
        
        video_input = frames_tensor.unsqueeze(0)  # Shape: [1, 16, 3, 224, 224]
        print(f"Processing batch {batch_idx + 1}/{num_batches}: video_input shape: {video_input.shape}")  # {{ edit_5: Debugging }}
        
        # **Ensure correct dimensions and channel count**
        assert video_input.dim() == 5, f"video_input should be 5D, but got {video_input.dim()}D"
        assert video_input.size(2) == expected_channels, f"Expected {expected_channels} channels, but got {video_input.size(2)}"
        
        # **Handle channel mismatch if any**
        if video_input.size(2) != expected_channels:
            logger.error(f"Channel mismatch: video_input has {video_input.size(2)} channels, expected {expected_channels}.")
            # Adjust channels if necessary
            if expected_channels == 1:
                # Convert RGB to Grayscale by averaging channels
                video_input = video_input.mean(dim=2, keepdim=True)
                print(f"Converted video_input to {video_input.size(2)} channels.")
            elif expected_channels == 4:
                # Add an alpha channel
                alpha_channel = torch.ones((video_input.size(0), video_input.size(1), 1, video_input.size(3), video_input.size(4)), device=device_type)
                video_input = torch.cat([video_input, alpha_channel], dim=2)
                print(f"Added alpha channel to video_input.")
            else:
                logger.error(f"Cannot adjust channels to {expected_channels}.")
                raise ValueError(f"Unsupported number of channels: {expected_channels}")
        
        # **Ensure correct dtype**
        video_input = video_input.to(torch.float32)
        
        # **Final shape check after adjustments**
        assert video_input.size(2) == expected_channels, f"After adjustment, expected {expected_channels} channels, but got {video_input.size(2)}"
        assert video_input.dim() == 5, f"After adjustment, video_input should be 5D, but got {video_input.dim()}D"
        
        # Process video batch through video encoder
        try:
            with torch.no_grad():
                batch_embeddings = video_encoder(video_input).last_hidden_state  # Shape: [1, num_tokens, hidden_size]
            print(f"Batch {batch_idx + 1} video_embeddings shape: {batch_embeddings.shape}")  # {{ edit_6: Debugging }}
            
            # Aggregate batch embeddings (e.g., mean pooling)
            batch_embeddings = batch_embeddings.mean(dim=1)  # Shape: [1, hidden_size]
            video_embeddings_list.append(batch_embeddings)  # {{ edit_7: Collect embeddings }}
        except Exception as e:
            logger.error(f"Failed to process batch {batch_idx + 1} through video_encoder: {e}")  # {{ edit_8 }}
            video_embeddings_list.append(torch.zeros((1, hidden_size), dtype=torch.float).to(device_type))  # {{ edit_9: Append dummy embedding }}
            print(f"Using dummy video embeddings for batch {batch_idx + 1} due to processing failure.")  # {{ edit_10 }}
    
    # Aggregate all batch embeddings (e.g., mean pooling across batches)
    video_embeddings = torch.mean(torch.stack(video_embeddings_list), dim=0)  # Shape: [1, hidden_size]
    print(f"Aggregated video_embeddings shape: {video_embeddings.shape}")  # {{ edit_11: Debugging }}
    
    return video_embeddings  # Return aggregated video embeddings
