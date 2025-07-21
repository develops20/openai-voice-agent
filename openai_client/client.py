# openai_client/client.py
import asyncio
import json
import requests
import keyboard
from aiortc import RTCPeerConnection, RTCSessionDescription
from config import Config
from audio.playback import AudioPlayback

# Conversation state variables
waiting_for_reply = False   # assistant is formulating / speaking
commit_received = False     # last VAD commit received, not yet queued

# Constants
MUTE_GRACE = 0.25    # seconds - let last speaker frames drain
VAD_WATCHDOG = 1.0   # seconds - force reply if server VAD is silent

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

async def handle_push_to_talk(mic_track, events_channel):
    """Handle push-to-talk functionality with spacebar"""
    global waiting_for_reply, commit_received
    
    loop = asyncio.get_running_loop()
    space_was_down = False
    mute_timer = None
    watchdog = None

    print("🎹 Push-to-talk ready! Press and hold SPACE to speak.")
    
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

                if not waiting_for_reply and mic_track.suspended:
                    mic_track.suspend(False)
                    print("🎙️ SPACE pressed → mic unmuted")

            # ── SPACE released (edge) ──────────────────────────────────────
            elif not space_down and space_was_down:
                if not mic_track.suspended:  # only schedule once
                    def _on_mute():
                        mic_track.suspend(True)
                        print("🔇 SPACE released → mic muted")
                        request_answer(mic_track, events_channel)  # normal path

                        # start watchdog in case VAD never fires
                        nonlocal watchdog
                        watchdog = loop.call_later(
                            VAD_WATCHDOG, check_missed_turn, mic_track, events_channel
                        )

                    mute_timer = loop.call_later(MUTE_GRACE, _on_mute)

            space_was_down = space_down
            await asyncio.sleep(0.02)  # 20 ms poll

        except Exception as e:
            print(f"⚠️ Push-to-talk error: {e}")
            await asyncio.sleep(0.1)

def check_missed_turn(mic_track, events_channel):
    """Force a response if server VAD never committed the turn."""
    global waiting_for_reply, commit_received
    
    if mic_track.suspended and not waiting_for_reply and not commit_received:
        print("⚠️ No VAD commit within 1s - forcing response.create")
        if events_channel and events_channel.readyState == "open":
            events_channel.send(json.dumps({"type": "response.create"}))
            waiting_for_reply = True

def request_answer(mic_track, events_channel):
    """
    Fire 'response.create' once both:
      • server finished VAD (commit_received)
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
                    "instructions": "You are a helpful assistant.",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.9,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": False  # Match reference code
                    }
                },
                timeout=20,
            )
            
            if response.status_code not in [200, 201]:
                print(f"❌ Failed to create session: {response.status_code} {response.text}")
                return False
                
            session_data = response.json()
            self.client_secret = session_data["client_secret"]["value"]
            self.session_id = session_data["id"]
            
            print("✅ OpenAI session created successfully")
            print(f"🆔 Session ID: {self.session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating session: {e}")
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
                
                # Send initial greeting
                self.events_channel.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message", 
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello! Ready to start voice conversation."}]
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
        global waiting_for_reply, commit_received
        
        try:
            event = json.loads(message)
            event_type = event.get("type", "")
            print(f"📨 Event: {event_type}")
            
            if event_type == "error":
                print(f"❌ OpenAI error: {json.dumps(event, indent=2)}")
            elif event_type == "input_audio_buffer.started":
                print("🎤 OpenAI started receiving audio")
            elif event_type == "input_audio_buffer.committed":
                print("✅ Audio buffer committed")
                commit_received = True
                request_answer(self.mic_track, self.events_channel)
            elif event_type == "response.output_item.added":
                print("💬 Assistant response starting")
                # Ensure mic is muted when assistant speaks
                if self.mic_track:
                    self.mic_track.suspend(True)
            elif event_type == "output_audio_buffer.started":
                print("🔊 Assistant audio output started")
            elif event_type == "response.audio_transcript.delta":
                # Don't print each delta to reduce noise, just show we're receiving
                if hasattr(self, '_last_transcript_print'):
                    if not self._last_transcript_print:
                        print("📝 Receiving audio transcript...")
                        self._last_transcript_print = True
                else:
                    print("📝 Receiving audio transcript...")
                    self._last_transcript_print = True
            elif event_type == "response.audio_transcript.done":
                print("📝 Audio transcript complete")
                self._last_transcript_print = False
            elif event_type == "response.done":
                print("✅ Response generation complete")
            elif event_type == "output_audio_buffer.stopped":
                print("🏁 Assistant finished speaking")
                # Reset conversation state
                waiting_for_reply = False
                commit_received = False
                print("🎙️ Ready for your next input (press SPACE to speak)")
                
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
        print("🛑 Disconnecting from OpenAI...")
        
        # Stop push-to-talk handler
        if self.push_to_talk_task:
            self.push_to_talk_task.cancel()
            try:
                await self.push_to_talk_task
            except asyncio.CancelledError:
                pass
        
        if self.player_task:
            self.player_task.cancel()
            try:
                await self.player_task
            except asyncio.CancelledError:
                pass
                
        self.audio_playback.stop_playback()
        
        if self.pc:
            await self.pc.close()
            print("🔌 WebRTC connection closed")
