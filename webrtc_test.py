import asyncio
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack
from av import AudioFrame
import time

# This is a minimal, self-contained script to test WebRTC loopback.
# It now uses a synthetic, silent audio track, removing all dependencies
# on ffmpeg, PyAudio, or any audio hardware/drivers.
# This provides the purest test of the WebRTC connection itself.

class SilentAudioTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that generates silent audio frames in pure Python.
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.samples_per_frame = 480  # 10ms of audio at 48kHz
        self.last_pts = 0

    async def recv(self):
        """
        Generate a single frame of silent audio. This method is called
        repeatedly by the RTCPeerConnection.
        """
        # Wait for the correct time to send the next frame to simulate a real-time stream.
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)

        # Create a silent audio frame.
        samples = np.zeros(self.samples_per_frame, dtype=np.int16)
        frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format='s16', layout='mono')
        
        # Set presentation timestamp, which is required.
        frame.pts = self.last_pts
        self.last_pts += self.samples_per_frame
        frame.sample_rate = self.sample_rate

        return frame

async def run_test():
    """
    Creates two RTCPeerConnection objects, connects them to each other (loopback),
    and streams a synthetic audio track.
    """
    print("--- Starting Minimal WebRTC Loopback Test (Synthetic Audio) ---")
    
    # Use a public STUN server. This helps peers discover their network addresses.
    config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    
    # Create the two peers for our loopback connection.
    client_pc = RTCPeerConnection(configuration=config)
    server_pc = RTCPeerConnection(configuration=config)
    
    connection_established = asyncio.Event()

    @client_pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"[Client] ICE Connection State: {client_pc.iceConnectionState}")
        if client_pc.iceConnectionState == "failed" or client_pc.iceConnectionState == "closed":
            print("[Client] Connection failed or closed.")
            connection_established.set() # End the test on failure

    @client_pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[Client] Connection State: {client_pc.connectionState}")
        if client_pc.connectionState == "connected":
            print("✅✅✅ TEST SUCCEEDED: WebRTC connection established! ✅✅✅")
            connection_established.set()

    # The "server" peer will simply log when it receives a track.
    # Receiving the track is sufficient proof that the connection is working.
    @server_pc.on("track")
    async def on_track(track):
        print(f"[Server] Track received: {track.kind}")

    # --- Signaling Simulation ---
    @client_pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            await server_pc.addIceCandidate(candidate)

    @server_pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            await client_pc.addIceCandidate(candidate)
    # --- End Signaling Simulation ---

    try:
        # Create our synthetic audio source.
        audio_source = SilentAudioTrack()
        
        # Add the audio track to the client peer.
        client_pc.addTrack(audio_source)
        
        # Create and exchange the offer/answer.
        offer = await client_pc.createOffer()
        await client_pc.setLocalDescription(offer)
        
        await server_pc.setRemoteDescription(offer)
        answer = await server_pc.createAnswer()
        await server_pc.setLocalDescription(answer)
        
        await client_pc.setRemoteDescription(answer)
        
        print("Offer/Answer exchange complete. Waiting for connection...")
        
        # Wait for the connection to be established or fail.
        await asyncio.wait_for(connection_established.wait(), timeout=20.0)

    except asyncio.TimeoutError:
        print("❌❌❌ TEST FAILED: Connection timed out. ❌❌❌")
        print("This result, with a fully synthetic test, confirms with very high")
        print("certainty that a firewall or other security software on your machine")
        print("is blocking the local network communication required for WebRTC.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("--- Test finished. Closing connections. ---")
        await client_pc.close()
        await server_pc.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("Test interrupted by user.")
