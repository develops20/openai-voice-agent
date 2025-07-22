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
        self.audio_capture = AudioCapture()
        self.openai_client = OpenAIRealtimeClient()
        self.is_running = False
        
    async def start(self):
        """Start the voice agent."""
        print("🤖 Starting Voice Agent...")
        
        if not Config.OPENAI_API_KEY:
            print("❌ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return
        
        try:
            # 1. Create OpenAI session first to get credentials
            session_created = await self.openai_client.create_session()
            if not session_created:
                print("❌ Failed to create OpenAI session - stopping.")
                return
            
            # 2. Start capturing audio from the microphone
            mic_track = await self.audio_capture.start_recording()
            
            # 3. Connect the WebRTC client using session credentials
            connection_success = await self.openai_client.connect(mic_track)
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
            
            # 5. The agent is now running
            self.is_running = True
            await self.conversation_loop()
            
        except Exception as e:
            print(f"❌ Error starting voice agent: {e}")
            await self.stop()
    
    async def conversation_loop(self):
        """
        Main loop to keep the agent running. Audio is streamed automatically
        in background tasks. This loop can be extended to handle text I/O
        or other agent logic.
        """
        print("🎙️  Voice Agent is running! Press Ctrl+C to stop.")
        while self.is_running:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
    
    async def stop(self):
        """Gracefully stop the voice agent and all its components."""
        print("🚨 DEBUG: VoiceAgent.stop() called - printing stack trace:")
        traceback.print_stack()
        
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

async def main():
    """Main entry point for the application."""
    agent = VoiceAgent()

    # For non-Windows platforms, set up signal handlers to gracefully stop the agent.
    # On Windows, add_signal_handler is not supported, and we rely on KeyboardInterrupt.
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        def signal_handler():
            if agent.is_running:
                print("\n🛑 Interrupt signal received. Shutting down...")
                # Schedule the stop coroutine to run on the event loop.
                loop.create_task(agent.stop())
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    try:
        await agent.start()
    except KeyboardInterrupt:
        # This will catch Ctrl+C on Windows. On other platforms, the signal
        # handler is preferred, but this provides a robust fallback.
        if agent.is_running:
            print("\n🛑 KeyboardInterrupt received. Shutting down...")
            await agent.stop()
    except asyncio.CancelledError:
        # This can happen if the main task is cancelled externally.
        if agent.is_running:
            await agent.stop()

if __name__ == "__main__":
    # asyncio.run() creates and manages the event loop.
    # The main() coroutine now handles its own exception for a clean shutdown.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This is a final fallback in case the interrupt happens during setup.
        print("\nProgram interrupted during startup.")
