import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from config import Config

class VoiceAgentPeerConnection:
    def __init__(self, audio_track):
        self.pc = RTCPeerConnection()
        self.audio_track = audio_track
        self.setup_peer_connection()
    
    def setup_peer_connection(self):
        """Setup WebRTC peer connection"""
        # Add audio track
        self.pc.addTrack(self.audio_track)
        
        # Setup event handlers
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"🔗 Connection state: {self.pc.connectionState}")
        
        @self.pc.on("track")
        def on_track(track):
            print(f"📡 Received track: {track.kind}")
    
    async def create_offer(self):
        """Create WebRTC offer"""
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        return offer
    
    async def close(self):
        """Close peer connection"""
        await self.pc.close()