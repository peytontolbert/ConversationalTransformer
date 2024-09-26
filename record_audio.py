import pyaudio
import wave
import time
import torch
import torchaudio
from transformers import WhisperModel, GPT2Tokenizer
import logging
import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize Whisper model
whisper_model = whisper.load_model("base")  # You can choose appropriate model size

text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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
