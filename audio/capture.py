# audio/capture.py
import asyncio
import numpy as np
import pyaudio
import threading
import traceback
from aiortc import MediaStreamTrack
from av import AudioFrame
from config import Config

class MicrophoneStreamTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that captures audio from the microphone using PyAudio.
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.frames_queue = asyncio.Queue()
        self.is_recording = False
        self.thread = None
        self.stream = None
        self.loop = asyncio.get_event_loop()
        
        # Push-to-talk state
        self.suspended = True  # Start suspended (muted)
        self.force_stop = False  # Flag to force stop recording
        
        # Timestamp tracking for proper PTS
        self.samples_sent = 0
        self.start_time = None
        
        # Audio level logging counter
        self.audio_level_counter = 0

    def suspend(self, suspended=True):
        """Suspend or resume the microphone track"""
        self.suspended = suspended
        status = "muted" if suspended else "unmuted"
        print(f"🎤 Microphone {status}")
        
        if not suspended:
            print("🎤 Ready for new audio input")

    def _start_recorder(self):
        """
        This method runs in a separate thread to manage the PyAudio stream,
        which is necessary because PyAudio's stream operations are blocking.
        """
        try:
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=Config.CHANNELS,
                rate=Config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=Config.CHUNK_SIZE,
                stream_callback=self._pyaudio_callback,
            )
            self.stream.start_stream()
            print("🎤 PyAudio recording started in a separate thread.")
            
            # Keep the thread alive while recording.
            # Only check self.is_recording, not stream.is_active() to avoid premature closure
            while self.is_recording and not self.force_stop:
                # Check if stream has an error and try to recover
                if not self.stream.is_active():
                    print("⚠️ PyAudio stream became inactive, attempting to restart...")
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                        
                        # Recreate the stream
                        self.stream = self.pyaudio_instance.open(
                            format=pyaudio.paInt16,
                            channels=Config.CHANNELS,
                            rate=Config.SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=Config.CHUNK_SIZE,
                            stream_callback=self._pyaudio_callback,
                        )
                        self.stream.start_stream()
                        print("✅ PyAudio stream restarted successfully")
                    except Exception as restart_error:
                        print(f"❌ Failed to restart PyAudio stream: {restart_error}")
                        break
                
                threading.Event().wait(0.1)

        except Exception as e:
            print(f"❌ PyAudio thread error: {e}")
        finally:
            # Only close if we're actually stopping, not due to stream becoming inactive
            if self.stream and (self.force_stop or not self.is_recording):
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    print("🎤 PyAudio stream closed.")
                except Exception as e:
                    print(f"⚠️ Error closing PyAudio stream: {e}")

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for the PyAudio stream. It's called by PyAudio in its own thread
        whenever new audio data is available.
        """
        # Check for stream errors
        if status:
            print(f"⚠️ PyAudio callback status: {status}")
        
        # Always capture audio, but queue it appropriately
        if not self.suspended and self.is_recording:
            # When not suspended, send audio immediately
            try:
                self.loop.call_soon_threadsafe(self.frames_queue.put_nowait, in_data)
                
                # Calculate and print audio level for debugging
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
                normalized_rms = rms / 32768.0
                
                # Log audio level occasionally to reduce spam
                self.audio_level_counter += 1
                if normalized_rms > 0.001 and self.audio_level_counter % 8 == 0:  # Every ~1 second at 8kHz
                    print(f"🎤 Live audio level: {normalized_rms:.4f}")
            except Exception as e:
                print(f"⚠️ Error in audio callback: {e}")
                
        return (None, pyaudio.paContinue)

    async def start(self):
        """
        Starts the recording by spawning a new thread for the PyAudio stream.
        """
        if not self.is_recording:
            self.is_recording = True
            self.force_stop = False
            self.thread = threading.Thread(target=self._start_recorder, daemon=True)
            self.thread.start()
            print("🎤 Recording thread started.")

    def stop(self):
        """
        Stops the recording thread and cleans up PyAudio resources.
        """
        if self.is_recording:
            print("🛑 Stopping audio recording...")
            self.is_recording = False
            self.force_stop = True
            
            if self.thread:
                self.thread.join(timeout=2)  # Wait for the thread to finish
                
            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                    print("🎤 PyAudio resources released.")
                except Exception as e:
                    print(f"⚠️ Error terminating PyAudio: {e}")

    async def recv(self):
        """
        This method is called by aiortc to get the next audio frame.
        It waits for a new frame to be available in the queue.
        """
        # Initialize start time on first call
        if self.start_time is None:
            import time
            self.start_time = time.time()
        
        try:
            # Try to get real audio data first
            try:
                data = await asyncio.wait_for(self.frames_queue.get(), timeout=0.02)
                self.frames_queue.task_done()
                
                # Create AudioFrame from real data
                audio_array = np.frombuffer(data, dtype=np.int16)
                reshaped = audio_array.reshape(Config.CHANNELS, -1)
                
                frame = AudioFrame.from_ndarray(
                    reshaped,
                    format='s16', 
                    layout='mono' if Config.CHANNELS == 1 else 'stereo'
                )
                frame.sample_rate = Config.SAMPLE_RATE
                
                # Set proper timestamp
                frame.pts = self.samples_sent
                self.samples_sent += Config.CHUNK_SIZE
                
                # Log occasionally to avoid spam - every ~2 seconds when speaking
                if self.samples_sent % (Config.CHUNK_SIZE * 16) == 0:  # Every ~2 seconds at 8kHz
                    print(f"🎤 Real audio frame: {Config.CHUNK_SIZE} samples, pts={frame.pts}")
                return frame
                
            except asyncio.TimeoutError:
                # Send silence frame with proper timestamp
                silence_data = np.zeros(Config.CHUNK_SIZE, dtype=np.int16)
                frame = AudioFrame.from_ndarray(
                    silence_data.reshape(Config.CHANNELS, -1),
                    format='s16', 
                    layout='mono' if Config.CHANNELS == 1 else 'stereo'
                )
                frame.sample_rate = Config.SAMPLE_RATE
                
                # Set proper timestamp
                frame.pts = self.samples_sent
                self.samples_sent += Config.CHUNK_SIZE
                
                # Only log occasionally to avoid spam
                if self.samples_sent % (Config.CHUNK_SIZE * 50) == 0:  # Every ~5 seconds at 8kHz
                    print(f"🔇 Silence frame: pts={frame.pts}")
                
                return frame
                
        except Exception as e:
            print(f"❌ Error in recv(): {e}")
            
            # Return silence on any exception
            try:
                silence_data = np.zeros(Config.CHUNK_SIZE, dtype=np.int16)
                frame = AudioFrame.from_ndarray(
                    silence_data.reshape(Config.CHANNELS, -1),
                    format='s16', 
                    layout='mono' if Config.CHANNELS == 1 else 'stereo'
                )
                frame.sample_rate = Config.SAMPLE_RATE
                
                # Set proper timestamp
                frame.pts = self.samples_sent
                self.samples_sent += Config.CHUNK_SIZE
                
                return frame
            except Exception as silence_error:
                print(f"💥 CRITICAL: Cannot create silence frame: {silence_error}")
                raise

class AudioCapture:
    """
    Manages the creation and lifecycle of the microphone audio track.
    """
    def __init__(self):
        self.track = None
        
    async def start_recording(self):
        """Create and start the microphone track."""
        self.track = MicrophoneStreamTrack()
        await self.track.start()
        print("🎤 Audio capture track created and started.")
        return self.track
    
    async def stop_recording(self):
        """Stop recording audio."""
        if self.track:
            self.track.stop()
            print("🎤 Audio capture track stopped.")
