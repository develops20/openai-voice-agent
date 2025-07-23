# audio/playback.py
import asyncio
import numpy as np
import pyaudio
from av import AudioFrame
from config import Config
from aiortc.mediastreams import MediaStreamError

class AudioPlayback:
    def __init__(self):
        self.is_playing = False
        self.pyaudio_instance = None
        self.output_stream = None
        
    def start_playback(self):
        """Initialize audio playback using PyAudio"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.output_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=Config.CHANNELS,
                rate=Config.SAMPLE_RATE,
                output=True,
                frames_per_buffer=Config.CHUNK_SIZE
            )
            print("🔊 Playback initialized")
            self.is_playing = True
        except Exception as e:
            print(f"❌ Error initializing playback: {e}")
    
    def stop_playback(self):
        """Stop audio playback and cleanup resources"""
        if self.is_playing:
            self.is_playing = False
            
            if self.output_stream:
                try:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                    print("🔊 Audio output stream closed")
                except Exception as e:
                    print(f"⚠️ Error closing audio stream: {e}")
                    
            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                    print("🔊 PyAudio playback terminated")
                except Exception as e:
                    print(f"⚠️ Error terminating PyAudio: {e}")
    
    async def play_track(self, track):
        """Play audio from an aiortc audio track"""
        print("🎵 Starting to play received audio track")
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        try:
            while self.is_playing:
                try:
                    # Receive audio frame from the track
                    frame = await track.recv()
                    consecutive_errors = 0  # Reset error count on successful frame
                    
                    # Convert AudioFrame to numpy array
                    audio_array = frame.to_ndarray()
                    
                    # Ensure it's the right format for PyAudio
                    if audio_array.dtype != np.int16:
                        # Normalize and convert to int16
                        audio_array = (audio_array * 32767).astype(np.int16)
                    
                    # Play the audio
                    if self.output_stream and self.is_playing:
                        audio_bytes = audio_array.tobytes()
                        self.output_stream.write(audio_bytes)
                        
                except MediaStreamError:
                    # Normal end-of-stream - OpenAI finished sending audio
                    print("✅ Audio stream ended normally")
                    break
                    
                except ConnectionError:
                    # Connection closed normally - also end-of-stream
                    print("✅ Audio connection closed normally") 
                    break
                    
                except (OSError, BrokenPipeError, ConnectionResetError):
                    # Network-related end-of-stream conditions
                    print("✅ Audio network connection ended normally")
                    break
                    
                except asyncio.TimeoutError:
                    # Timeout waiting for next frame - normal end
                    print("✅ Audio stream timeout (normal end)")
                    break
                    
                except StopAsyncIteration:
                    # Async iterator exhausted - normal end
                    print("✅ Audio stream iteration complete")
                    break
                    
                except Exception as frame_error:
                    # Check if it's an end-of-stream related error by message content
                    error_msg = str(frame_error).lower()
                    if any(phrase in error_msg for phrase in [
                        "connection closed", "stream ended", "eof", "end of file", 
                        "no more data", "stream closed", "track ended"
                    ]):
                        print(f"✅ Audio stream ended normally: {frame_error}")
                        break
                    
                    # Real errors - count these toward limit
                    consecutive_errors += 1
                    print(f"⚠️ Audio frame error ({consecutive_errors}/{max_consecutive_errors}): {frame_error}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("❌ Too many consecutive audio errors, stopping playback")
                        break
                    
                    # Brief pause before retrying
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Error playing track: {e}")
        finally:
            print("🔊 Track playback stopped")