# audio/capture.py
import asyncio
import numpy as np
import pyaudio
import threading
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
        
        # Suspend functionality for push-to-talk
        self.suspended = True  # Start suspended (muted)

    def suspend(self, suspended=True):
        """Suspend or resume the microphone track"""
        self.suspended = suspended
        status = "muted" if suspended else "unmuted"
        print(f"🎤 Microphone {status}")

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
            while self.is_recording and self.stream.is_active():
                # A short sleep prevents this loop from consuming too much CPU.
                threading.Event().wait(0.1)

        except Exception as e:
            print(f"❌ PyAudio thread error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            print("🎤 PyAudio stream closed.")

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for the PyAudio stream. It's called by PyAudio in its own thread
        whenever new audio data is available.
        """
        # Only queue data if not suspended
        if not self.suspended:
            self.loop.call_soon_threadsafe(self.frames_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    async def start(self):
        """
        Starts the recording by spawning a new thread for the PyAudio stream.
        """
        if not self.is_recording:
            self.is_recording = True
            self.thread = threading.Thread(target=self._start_recorder, daemon=True)
            self.thread.start()
            print("🎤 Recording thread started.")

    def stop(self):
        """
        Stops the recording thread and cleans up PyAudio resources.
        """
        if self.is_recording:
            self.is_recording = False
            if self.thread:
                self.thread.join(timeout=1) # Wait for the thread to finish
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            print("🎤 PyAudio resources released.")

    async def recv(self):
        """
        This method is called by aiortc to get the next audio frame.
        It waits for a new frame to be available in the queue.
        """
        # If suspended, send silence
        if self.suspended:
            # Generate silence frame
            silence_data = np.zeros(Config.CHUNK_SIZE, dtype=np.int16)
            frame = AudioFrame.from_ndarray(
                silence_data.reshape(-1, Config.CHANNELS),
                format='s16', 
                layout='mono' if Config.CHANNELS == 1 else 'stereo'
            )
            frame.sample_rate = Config.SAMPLE_RATE
            frame.pts = None
            return frame
        
        # Wait for real audio data when not suspended
        try:
            data = await asyncio.wait_for(self.frames_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            # Send silence if no data available
            silence_data = np.zeros(Config.CHUNK_SIZE, dtype=np.int16)
            frame = AudioFrame.from_ndarray(
                silence_data.reshape(-1, Config.CHANNELS),
                format='s16', 
                layout='mono' if Config.CHANNELS == 1 else 'stereo'
            )
            frame.sample_rate = Config.SAMPLE_RATE
            frame.pts = None
            return frame
        
        # Create an AudioFrame from the raw PCM data.
        frame = AudioFrame.from_ndarray(
            np.frombuffer(data, dtype=np.int16).reshape(-1, Config.CHANNELS),
            format='s16', 
            layout='mono' if Config.CHANNELS == 1 else 'stereo'
        )
        frame.sample_rate = Config.SAMPLE_RATE
        frame.pts = None # Let aiortc handle the presentation timestamp

        self.frames_queue.task_done()
        return frame

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
            self.track.stop()  # Now synchronous, no await needed
            print("🎤 Audio capture track stopped.")
