# main.py
import asyncio
import signal
import sys
import traceback
from audio.capture import AudioCapture
from openai_client.client import OpenAIRealtimeClient
from config import Config

class VoiceAgent:
    def __init__(self):
        self.audio_capture = None
        self.openai_client = None
        self.is_running = False

    async def start(self):
        """Initialize and start the voice agent."""
        print("🚀 Starting Voice Agent...")
        
        # Check API key first
        if not Config.OPENAI_API_KEY:
            print("❌ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return
        
        try:
            # Initialize audio capture
            self.audio_capture = AudioCapture()
            
            # Initialize OpenAI client
            self.openai_client = OpenAIRealtimeClient()
            
            # 1. Create OpenAI session first to get credentials
            session_created = await self.openai_client.create_session()
            if not session_created:
                print("❌ Failed to create OpenAI session - stopping.")
                return
            
            # 2. Start capturing audio from the microphone
            track = await self.audio_capture.start_recording()
            
            # 3. Connect the WebRTC client using session credentials
            connection_success = await self.openai_client.connect(track)
            if not connection_success:
                print("❌ Failed to connect to OpenAI - stopping.")
                await self.stop()
                return
            
            # 4. Wait for the WebRTC session to be established
            session_ready = await self.openai_client.wait_for_session()
            if not session_ready:
                print("❌ Failed to establish session - stopping.")
                await self.stop()
                return
            
            self.is_running = True
            print("✅ Voice Agent started successfully!")
            
            # Keep the agent running
            try:
                while self.is_running:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                
        except Exception as e:
            print(f"❌ Error starting Voice Agent: {e}")
            traceback.print_exc()
            raise

    async def stop(self):
        """Gracefully stop the voice agent and all its components."""
        # Check is_running to prevent stop from being called multiple times.
        if self.is_running:
            print("\n🛑 Stopping Voice Agent...")
            self.is_running = False

            # Stop audio capture
            await self.audio_capture.stop_recording()

            # Disconnect the client (which also stops playback)
            if self.openai_client:
                await self.openai_client.disconnect()

            print("✅ Voice Agent stopped.")

async def signal_handler():
    """Handle shutdown signals gracefully."""
    print("\n🔄 Graceful shutdown initiated...")
    # The main loop will handle cleanup

async def main():
    agent = VoiceAgent()
    
    # Setup signal handlers for graceful shutdown
    def handle_signal():
        asyncio.create_task(agent.stop())
    
    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: handle_signal())
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        print("\n🔄 Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
    finally:
        await agent.stop()
        print("👋 Voice Agent terminated.")

if __name__ == "__main__":
    print("🎤 Voice Agent with OpenAI Realtime API")
    print("📝 Controls:")
    print("   - Hold SPACEBAR to talk")
    print("   - Release SPACEBAR when done speaking")
    print("   - Press Ctrl+C to quit")
    print("=" * 50)
    
    asyncio.run(main())
