#!/usr/bin/env python3
"""
Audio Test Script - Test microphone capture and audio levels
Run this to verify your microphone is working before testing the voice agent.
"""

import pyaudio
import numpy as np
import time
import threading
from config import Config

class AudioTester:
    def __init__(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.is_recording = False
        
    def test_microphone(self, duration=10):
        """Test microphone for the specified duration"""
        print(f"🎤 Testing microphone for {duration} seconds...")
        print(f"📊 Sample rate: {Config.SAMPLE_RATE}Hz, Channels: {Config.CHANNELS}")
        print(f"🔊 Speak into your microphone now!\n")
        
        try:
            stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=Config.CHANNELS,
                rate=Config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=Config.CHUNK_SIZE,
            )
            
            start_time = time.time()
            max_level = 0
            total_samples = 0
            
            while time.time() - start_time < duration:
                try:
                    data = stream.read(Config.CHUNK_SIZE, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate RMS (Root Mean Square) for audio level
                    rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
                    normalized_rms = rms / 32768.0
                    
                    max_level = max(max_level, normalized_rms)
                    total_samples += 1
                    
                    # Create a visual level meter
                    level_bars = int(normalized_rms * 50)  # Scale to 50 chars
                    level_meter = "█" * level_bars + "░" * (50 - level_bars)
                    
                    # Only update display if there's significant audio
                    if normalized_rms > 0.001:
                        print(f"\r🎤 Audio Level: [{level_meter}] {normalized_rms:.4f}", end="", flush=True)
                    
                    time.sleep(0.05)  # Update every 50ms
                    
                except Exception as e:
                    print(f"\n⚠️ Error reading audio: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
            print(f"\n\n✅ Test completed!")
            print(f"📊 Maximum audio level detected: {max_level:.4f}")
            
            if max_level > 0.01:
                print("✅ Microphone is working well - good audio levels detected")
            elif max_level > 0.001:
                print("⚠️ Microphone detected but audio levels are low - speak louder or adjust mic settings")
            else:
                print("❌ No audio detected - check microphone connection and permissions")
                
        except Exception as e:
            print(f"❌ Error testing microphone: {e}")
            print("💡 Make sure your microphone is connected and not used by other applications")
        
    def list_audio_devices(self):
        """List available audio input devices"""
        print("🎤 Available audio input devices:")
        device_count = self.pyaudio_instance.get_device_count()
        
        for i in range(device_count):
            device_info = self.pyaudio_instance.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"  Device {i}: {device_info['name']} (Channels: {device_info['maxInputChannels']})")
        
    def cleanup(self):
        """Cleanup PyAudio resources"""
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()

if __name__ == "__main__":
    tester = AudioTester()
    
    try:
        print("🎵 Voice Agent Audio Test\n")
        
        # List available devices
        tester.list_audio_devices()
        print()
        
        # Test microphone
        tester.test_microphone(duration=10)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        tester.cleanup()
        print("🔧 Audio test finished") 