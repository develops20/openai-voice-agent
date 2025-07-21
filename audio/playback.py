# audio/playback.py
import asyncio
import numpy as np
import queue
import pyaudio
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from av import AudioFrame
from config import Config

class AudioPlayback:
    def __init__(self):
        self.media_player = None
        self.playback_queue = asyncio.Queue()
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
                        
                except Exception as frame_error:
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
    
    def play_audio(self, audio_data):
        """Add audio data to playback queue"""
        if self.is_playing:
            print(f"🎵 Queued {len(audio_data)} bytes for playback.")
            self.playback_queue.put_nowait(audio_data)

    async def process_playback_queue(self, player_track):
        """Process audio playback queue using aiortc"""
        if not player_track:
            await asyncio.sleep(0.01)
            return
            
        while self.is_playing:
            try:
                audio_data = await self.playback_queue.get()
                # Convert to numpy array if needed
                if isinstance(audio_data, bytes):
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                else:
                    audio_array = audio_data
                # Ensure it's a 1D array for mono
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.flatten()

                # Create an AudioFrame
                frame = AudioFrame.from_ndarray(
                    np.reshape(audio_array, (-1, 1)), # Reshape to (samples, channels)
                    format='s16', 
                    layout='mono'
                )
                frame.sample_rate = Config.SAMPLE_RATE
                frame.pts = None # Let aiortc handle pts

                await player_track.send(frame)
                # print(f"   -> Sent audio frame to player track.")
                self.playback_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error in playback processing: {e}")
                # Let's not break the loop for a single error
                await asyncio.sleep(0.01)
    
    def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        print("🔊 Playback stopped")