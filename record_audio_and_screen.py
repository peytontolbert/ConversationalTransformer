import threading
import torch
import logging
from record_video import record_screen
from transformers import GPT2LMHeadModel, VideoMAEModel
from record_audio import record_audio
logger = logging.getLogger(__name__)

device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device_type)
hidden_size = text_model.config.hidden_size
# Device configuration

video_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device_type)
def record_audio_and_screen(audio_filename, screen_filename, duration, desired_num_frames=160):
    audio_result = {}
    screen_result = {}

    def audio_thread():
        audio_tensor, audio_text_inputs = record_audio(audio_filename, duration)
        audio_result['audio_tensor'] = audio_tensor
        audio_result['audio_text_inputs'] = audio_text_inputs

    def screen_thread():
        try:
            # Directly use video_embeddings returned from record_screen without permutation
            video_embeddings = record_screen(screen_filename, duration, desired_num_frames)
            screen_result['video_embeddings'] = video_embeddings
        except Exception as e:
            logger.error(f"Failed to record screen: {e}")  # {{ edit_5 }}
            # Create dummy video embeddings in case of failure
            video_embeddings = torch.zeros((1, hidden_size), dtype=torch.float).to(device_type)
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
