# openai_client/client.py
import asyncio
import json
import requests
import keyboard
from aiortc import RTCPeerConnection, RTCSessionDescription
from config import Config
from audio.playback import AudioPlayback
import os
import logging
from datetime import datetime

# Set up logging for OpenAI voice session
LOG_DIR = "logs"
LOG_FILE = "openai_voice_session.log"

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Set up file logger
log_file_path = os.path.join(LOG_DIR, LOG_FILE)
file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Create logger
session_logger = logging.getLogger('openai_voice_session')
session_logger.setLevel(logging.INFO)
session_logger.addHandler(file_handler)

# Prevent duplicate logs
session_logger.propagate = False

# Log session start
session_logger.info("=" * 60)
session_logger.info("NEW VOICE SESSION STARTED")
session_logger.info("=" * 60)

# Conversation state variables
waiting_for_reply = False
commit_received = False
user_transcript = ""

# Push-to-talk timing constants
MUTE_GRACE = 0.1  # seconds to wait before muting after release
VAD_WATCHDOG = 3.0  # seconds to wait for VAD before assuming it missed the audio

# Additional state for mute timing
mute_grace_start = None

def prefer_audio_codec(pc, codec_name):
    """Prefer a specific audio codec for WebRTC connection"""
    try:
        # Get the first audio transceiver
        transceivers = pc.getTransceivers()
        audio_transceiver = next((t for t in transceivers if t.kind == "audio"), None)
        
        if audio_transceiver and hasattr(audio_transceiver, 'setCodecPreferences'):
            # Get available codecs
            codecs = audio_transceiver.sender.getCapabilities('audio').codecs
            
            # Find the preferred codec
            preferred_codec = next(
                (c for c in codecs if codec_name.lower() in c.mimeType.lower()), 
                None
            )
            
            if preferred_codec:
                # Move preferred codec to front
                codecs.remove(preferred_codec)
                codecs.insert(0, preferred_codec)
                audio_transceiver.setCodecPreferences(codecs)
                print(f"🎵 Preferred codec set to: {codec_name}")
            else:
                print(f"⚠️ Codec {codec_name} not found in capabilities")
        else:
            print("⚠️ Audio transceiver not found or codec preferences not supported")
    except Exception as e:
        print(f"⚠️ Could not set codec preference: {e}")

def check_missed_turn(mic_track, events_channel):
    """Check if VAD missed the audio and push-to-talk needs to force a response"""
    global waiting_for_reply, commit_received
    
    # Only intervene if we're still waiting and nothing has happened
    if commit_received and not waiting_for_reply:
        print("⚠️ VAD timeout - forcing response.create (OpenAI didn't detect speech automatically)")
        session_logger.warning("VAD: Timeout occurred - forcing response.create")
        request_answer(mic_track, events_channel)
    else:
        # Audio was already processed successfully, no need to intervene
        session_logger.info("VAD: Timeout occurred but audio was already processed successfully")

async def handle_push_to_talk(mic_track, events_channel):
    """Handle push-to-talk functionality with spacebar"""
    global waiting_for_reply, commit_received, user_transcript
    
    loop = asyncio.get_running_loop()
    space_was_down = False
    mute_timer = None
    watchdog = None
    speech_start_time = None
    
    print("🎹 Push-to-talk ready! Press and hold SPACE to speak.")
    session_logger.info("SYSTEM: Push-to-talk ready")
    
    while True:
        try:
            space_down = keyboard.is_pressed("space")
            
            # ── SPACE pressed (edge) ───────────────────────────────────────
            if space_down and not space_was_down:
                # Cancel pending timers (user resumed speaking)
                if mute_timer:
                    mute_timer.cancel()
                    mute_timer = None
                if watchdog:
                    watchdog.cancel()
                    watchdog = None

                if not waiting_for_reply:
                    mic_track.suspend(False)
                    speech_start_time = loop.time()
                    user_transcript = ""  # Reset transcript for new input
                    print("🎙️ SPACE pressed → Recording your speech...")
                    print("🎤 Speak now! Release SPACE when done.")
                    session_logger.info("USER_INPUT: Started speaking (spacebar pressed)")

            # ── SPACE released (edge) ──────────────────────────────────────
            elif not space_down and space_was_down:
                if not mic_track.suspended and speech_start_time:  
                    speech_duration = loop.time() - speech_start_time
                    print(f"🔇 SPACE released → Speech recorded ({speech_duration:.1f}s)")
                    session_logger.info(f"USER_INPUT: Stopped speaking (spacebar released) - Duration: {speech_duration:.1f}s")
                    
                    def _on_mute():
                        mic_track.suspend(True)
                        print("🔇 Microphone muted - processing your speech...")
                        
                        # Send input_audio_buffer.commit to signal end of input
                        if events_channel and events_channel.readyState == "open":
                            events_channel.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            print("📤 Audio buffer committed to OpenAI")
                            session_logger.info("USER_INPUT: Audio buffer committed to OpenAI")

                        # Start watchdog in case VAD doesn't process the audio
                        nonlocal watchdog
                        watchdog = loop.call_later(
                            VAD_WATCHDOG, check_missed_turn, mic_track, events_channel
                        )

                    mute_timer = loop.call_later(MUTE_GRACE, _on_mute)
                    speech_start_time = None

            space_was_down = space_down
            await asyncio.sleep(0.05)  # Check every 50ms
            
        except Exception as e:
            error_msg = f"Push-to-talk error: {e}"
            print(f"⚠️ {error_msg}")
            session_logger.error(f"PUSH_TO_TALK: {error_msg}")
            await asyncio.sleep(0.1)

def request_answer(mic_track, events_channel):
    """
    Fire 'response.create' once both:
      • audio buffer committed (commit_received)
      • mic is muted (user released SPACE)
    """
    global waiting_for_reply, commit_received

    print(f"🔔 Checking conditions: suspended={mic_track.suspended}, commit={commit_received}, waiting={waiting_for_reply}")

    if mic_track.suspended and commit_received and not waiting_for_reply:
        print("🔔 Conditions met - sending response.create")
        if events_channel and events_channel.readyState == "open":
            events_channel.send(json.dumps({"type": "response.create"}))
            waiting_for_reply = True
            commit_received = False

class OpenAIRealtimeClient:
    def __init__(self):
        self.pc = None
        self.client_secret = None
        self.session_id = None
        self.events_channel = None
        self.audio_playback = AudioPlayback()
        self.player_task = None
        self.is_connected = False
        self.mic_track = None
        self.push_to_talk_task = None
        
    async def create_session(self):
        """Step 1: Create session with OpenAI API to get credentials"""
        print("📞 Creating OpenAI Realtime session...")
        session_logger.info("SESSION: Creating OpenAI Realtime session")
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": Config.OPENAI_MODEL,
                    "voice": Config.OPENAI_VOICE,
                    "instructions": "You are a helpful assistant. Keep responses concise and natural.",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 2000,  # Longer silence to reduce interference with push-to-talk
                        "create_response": False  # Don't auto-create responses - let push-to-talk handle it
                    },
                    "input_audio_format": "g711_ulaw",  # PCMU format for WebRTC compatibility  
                    "output_audio_format": "g711_ulaw"  # PCMU format for WebRTC compatibility
                },
                timeout=20,
            )
            
            if response.status_code not in [200, 201]:
                error_msg = f"Failed to create session: {response.status_code} {response.text}"
                print(f"❌ {error_msg}")
                session_logger.error(f"SESSION: {error_msg}")
                return False
                
            session_data = response.json()
            self.client_secret = session_data["client_secret"]["value"]
            self.session_id = session_data["id"]
            
            print("✅ OpenAI session created successfully")
            print(f"🆔 Session ID: {self.session_id}")
            
            session_logger.info(f"SESSION: Created successfully - ID: {self.session_id}")
            session_logger.info(f"SESSION_RESPONSE: {json.dumps(session_data, indent=2)}")
            return True
            
        except Exception as e:
            error_msg = f"Error creating session: {e}"
            print(f"❌ {error_msg}")
            session_logger.error(f"SESSION: {error_msg}")
            return False

    async def connect(self, microphone_track):
        """Step 2: Create WebRTC connection using session credentials"""
        print("🔗 Setting up WebRTC connection to OpenAI...")
        
        try:
            # Store mic track reference for push-to-talk
            self.mic_track = microphone_track
            
            # Create peer connection
            self.pc = RTCPeerConnection()
            
            # Add microphone track
            self.pc.addTrack(microphone_track)
            print("🎤 Added microphone track")
            
            # Prefer PCMU codec (important for OpenAI compatibility)
            prefer_audio_codec(self.pc, "pcmu")
            
            # Set up data channel for events BEFORE creating offer
            self.events_channel = self.pc.createDataChannel("oai-events")
            
            @self.events_channel.on("open")
            def on_events_open():
                print("📡 Events channel opened")
                # Start push-to-talk handler
                self.push_to_talk_task = asyncio.create_task(
                    handle_push_to_talk(self.mic_track, self.events_channel)
                )
                
                # Send initial greeting to start conversation
                self.events_channel.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message", 
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello! I'm ready to have a voice conversation with you."}]
                    }
                }))
                self.events_channel.send(json.dumps({"type": "response.create"}))
                
                global waiting_for_reply
                waiting_for_reply = True
            
            @self.events_channel.on("message")
            def on_events_message(message):
                self._handle_event(message)
            
            # Set up audio track handling
            @self.pc.on("track")
            async def on_track(track):
                print(f"📡 Received audio track: {track.kind}")
                if track.kind == "audio":
                    print("🔊 Starting audio playback...")
                    self.audio_playback.start_playback()
                    self.player_task = asyncio.create_task(self.audio_playback.play_track(track))
            
            @self.pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"🔗 Connection state: {self.pc.connectionState}")
                if self.pc.connectionState == "connected":
                    self.is_connected = True
                    print("✅ WebRTC connection established!")
                elif self.pc.connectionState == "failed":
                    print("❌ WebRTC connection failed!")
            
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            print("📤 Sending SDP offer to OpenAI...")
            print(f"📝 Local SDP:\n{self.pc.localDescription.sdp}")
            
            # Send SDP to OpenAI - Note: expects 201 response, not 200
            response = requests.post(
                f"https://api.openai.com/v1/realtime?model={Config.OPENAI_MODEL}",
                headers={
                    "Authorization": f"Bearer {self.client_secret}",
                    "Content-Type": "application/sdp"
                },
                data=self.pc.localDescription.sdp,
                timeout=20
            )
            
            # Check for success (201 is expected for SDP exchange)
            if response.status_code not in [200, 201]:
                print(f"❌ SDP exchange failed: {response.status_code} {response.text}")
                return False
            
            # Set remote description
            answer_sdp = response.text
            print(f"📝 Remote SDP:\n{answer_sdp}")
            
            await self.pc.setRemoteDescription(RTCSessionDescription(answer_sdp, "answer"))
            
            print("✅ SDP exchange completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to OpenAI: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _handle_event(self, message):
        """Handle events from OpenAI"""
        global waiting_for_reply, commit_received, user_transcript
        
        try:
            event = json.loads(message)
            event_type = event.get("type", "")
            
            # Log all events to file
            session_logger.info(f"EVENT: {event_type}")
            session_logger.info(f"EVENT_DATA: {json.dumps(event, indent=2)}")
            
            # Reduce noise from repetitive events
            if event_type in ["response.audio_transcript.delta", "conversation.item.input_audio_transcription.delta"]:
                # Don't log these repetitive events individually
                pass
            else:
                print(f"📨 Event: {event_type}")
            
            if event_type == "error":
                error_msg = f"OpenAI error: {json.dumps(event, indent=2)}"
                print(f"❌ {error_msg}")
                session_logger.error(error_msg)
            elif event_type == "input_audio_buffer.started":
                msg = "OpenAI started receiving your audio"
                print(f"🎤 {msg}")
                session_logger.info(f"AUDIO_INPUT: {msg}")
            elif event_type == "input_audio_buffer.committed":
                msg = "Your audio was successfully received by OpenAI"
                print(f"✅ {msg}")
                session_logger.info(f"AUDIO_INPUT: {msg}")
                commit_received = True
                request_answer(self.mic_track, self.events_channel)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                # This shows what OpenAI heard from the user
                transcript = event.get("transcript", "")
                if transcript.strip():
                    msg = f'You said: "{transcript}"'
                    print(f"👤 {msg}")
                    session_logger.info(f"USER_TRANSCRIPT: {transcript}")
                    user_transcript = transcript
                else:
                    msg = "(OpenAI detected speech but couldn't transcribe it clearly)"
                    print(f"👤 {msg}")
                    session_logger.warning(f"USER_TRANSCRIPT: Failed to transcribe clearly")
            elif event_type == "conversation.item.input_audio_transcription.failed":
                msg = "OpenAI couldn't transcribe your speech - try speaking more clearly"
                print(f"⚠️ {msg}")
                session_logger.warning(f"USER_TRANSCRIPT: {msg}")
            elif event_type == "response.output_item.added":
                msg = "Assistant is preparing response..."
                print(f"💬 {msg}")
                session_logger.info(f"ASSISTANT: {msg}")
                # Ensure mic is muted when assistant speaks
                if self.mic_track:
                    self.mic_track.suspend(True)
            elif event_type == "output_audio_buffer.started":
                msg = "Assistant is speaking..."
                print(f"🔊 {msg}")
                session_logger.info(f"ASSISTANT: {msg}")
            elif event_type == "response.audio_transcript.delta":
                # Don't print each delta to reduce noise
                if not hasattr(self, '_receiving_transcript'):
                    print("📝 Receiving assistant response...")
                    session_logger.info("ASSISTANT: Starting to receive response transcript")
                    self._receiving_transcript = True
            elif event_type == "response.audio_transcript.done":
                msg = "Assistant response complete"
                print(f"📝 {msg}")
                session_logger.info(f"ASSISTANT: {msg}")
                if hasattr(self, '_receiving_transcript'):
                    del self._receiving_transcript
            elif event_type == "response.done":
                msg = "Response generation complete"
                print(f"✅ {msg}")
                session_logger.info(f"ASSISTANT: {msg}")
            elif event_type == "output_audio_buffer.stopped":
                msg = "Assistant finished speaking"
                print(f"🏁 {msg}")
                session_logger.info(f"ASSISTANT: {msg}")
                # Reset conversation state
                waiting_for_reply = False
                commit_received = False
                print("🎙️ Ready for your next input (press and hold SPACE to speak)")
                session_logger.info("SYSTEM: Ready for next user input")
                
        except Exception as e:
            print(f"⚠️ Error handling event: {e}")

    async def wait_for_session(self):
        """Wait for connection to be established"""
        print("⏳ Waiting for WebRTC connection...")
        
        # Wait for connection
        for i in range(30):  # 30 second timeout
            if self.is_connected:
                print("✅ Session ready!")
                return True
            await asyncio.sleep(1)
        
        print("❌ Connection timeout!")
        return False

    async def disconnect(self):
        """Clean up and disconnect"""
        session_logger.info("SESSION: Disconnecting from OpenAI")
        print("🛑 Disconnecting from OpenAI...")
        
        # Stop push-to-talk handler
        if hasattr(self, 'push_to_talk_task') and self.push_to_talk_task:
            self.push_to_talk_task.cancel()
            try:
                await self.push_to_talk_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, 'player_task') and self.player_task:
            self.player_task.cancel()
            try:
                await self.player_task
            except asyncio.CancelledError:
                pass
                
        if self.audio_playback:
            self.audio_playback.stop_playback()
            session_logger.info("SESSION: Audio playback stopped")
        
        if self.pc:
            await self.pc.close()
            print("🔌 WebRTC connection closed")
            session_logger.info("SESSION: WebRTC connection closed")
        
        session_logger.info("SESSION: Disconnection complete")
        session_logger.info("=" * 60)
        session_logger.info("VOICE SESSION ENDED")
        session_logger.info("=" * 60)
