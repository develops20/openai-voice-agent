import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-realtime-preview')
    OPENAI_VOICE = os.getenv('OPENAI_VOICE', 'alloy')
    
    # Audio Configuration
    SAMPLE_RATE = 8000  # PCMU codec uses 8kHz
    CHUNK_SIZE = 1024
    CHANNELS = 1
    AUDIO_FORMAT = 'int16'
    
    # WebRTC Configuration
    STUN_SERVER = 'stun:stun.l.google.com:19302'
    
    # Voice Activity Detection
    VAD_THRESHOLD = 0.001  # Lowered from 0.01 to be more sensitive
    SILENCE_DURATION = 1.0  # seconds of silence before stopping recording