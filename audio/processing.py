import numpy as np
from config import Config

class AudioProcessor:
    def __init__(self):
        self.silence_counter = 0
        self.silence_threshold = int(Config.SILENCE_DURATION * Config.SAMPLE_RATE / Config.CHUNK_SIZE)
    
    def detect_voice_activity(self, audio_data):
        """Simple voice activity detection"""
        if isinstance(audio_data, np.ndarray):
            try:
                # Fix: Handle potential invalid values in audio data
                valid_audio = audio_data[np.isfinite(audio_data)]
                if len(valid_audio) == 0:
                    return False
                    
                # Ensure we don't have negative values that could cause sqrt issues
                squared_audio = valid_audio**2
                if not np.all(squared_audio >= 0):
                    return False
                    
                rms = np.sqrt(np.mean(squared_audio))
                normalized_rms = rms / 32768.0  # Normalize for int16
                print(f"🎤 Mic level: {normalized_rms:.4f} | VAD Threshold: {Config.VAD_THRESHOLD}")
                
                if normalized_rms > Config.VAD_THRESHOLD:
                    self.silence_counter = 0
                    return True
                else:
                    self.silence_counter += 1
                    return self.silence_counter < self.silence_threshold
            except Exception as e:
                print(f"🎤 VAD error: {e}")
                return False
        return False
    
    def convert_to_openai_format(self, audio_data):
        """Convert audio data to OpenAI API format"""
        if isinstance(audio_data, np.ndarray):
            # Ensure it's int16 format
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            return audio_data.tobytes()
        return audio_data
    
    def convert_from_openai_format(self, audio_bytes):
        """Convert OpenAI API response to numpy array"""
        return np.frombuffer(audio_bytes, dtype=np.int16)