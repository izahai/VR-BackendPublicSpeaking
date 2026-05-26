import asyncio
import json
import sys
import os
import requests
import numpy as np
import sounddevice as sd
import websockets

# Configurations
HTTP_URL = "http://127.0.0.1:8000/api/stt_upload"
WS_URL = "ws://127.0.0.1:8000/ws/audio_similarity"

SAMPLE_RATE = 16000  # Target Whisper sample rate
CHANNELS = 1         # Mono audio
BLOCK_DURATION = 0.5 # Window segment slice duration in seconds

# Dynamic Global state
teleprompter_lines = []
current_highlighted_idx = -1
scroll_offset = 0    # Tracks extra scrolling shifts based on similarity hits

def fetch_teleprompter_script():
    """Fetches the formatted script text from the FastAPI endpoint."""
    global teleprompter_lines
    print(f"📡 Fetching target script from {HTTP_URL}...")
    try:
        response = requests.get(HTTP_URL)
        if response.status_code == 200:
            data = response.json()
            raw_txt = data.get("format_txt", "")
            # Split lines and drop empty strings
            teleprompter_lines = [line.strip() for line in raw_txt.split("\n") if line.strip()]
            print(f"✅ Successfully loaded {len(teleprompter_lines)} tracking lines.")
            return data.get("cur_idx_cluster", 0)
        else:
            print(f"⚠️ Failed to get script. Server responded with: {response.status_code}")
    except Exception as e:
        print(f"❌ Error communicating with REST API: {e}")
    
    print("Falling back to empty script tracking.")
    return 0

def render_ui(latest_transcription, similarity):
    """Clears the terminal and renders a visual tracker dashboard."""
    global scroll_offset
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 65)
    print("  🎤 MACBOOK TELEPROMPTER TRACKING STREAM (Ctrl+C to Stop) ")
    print("=" * 65)
    print(f"🗣️ Last Heard : {latest_transcription}")
    print(f"🎯 Similarity Score : {similarity}")
    print("-" * 65)
    print("📖 SCRIPT WINDOW:")
    
    focus_idx = max(0, current_highlighted_idx) + scroll_offset
    
    if current_highlighted_idx == -1 and scroll_offset == 0:
        start_view = 0
        end_view = min(len(teleprompter_lines), 14)
    else:
        start_view = max(0, focus_idx - 6)
        end_view = min(len(teleprompter_lines), start_view + 14)
        
        if end_view == len(teleprompter_lines):
            start_view = max(0, end_view - 14)
            
    if not teleprompter_lines:
        print("  [No script text loaded from endpoint]")
    
    for idx, line in enumerate(teleprompter_lines):
        if idx < start_view or idx >= end_view:
            continue
            
        if idx == current_highlighted_idx:
            print(f" 👉 \033[1;32;40m[{idx:02d}] {line}\033[0m")
        else:
            print(f"    [{idx:02d}] {line}")
            
    print("=" * 65)

async def stream_mic_audio():
    global current_highlighted_idx, scroll_offset
    
    # Step 1: Initial REST sync setup
    fetch_teleprompter_script()
    await asyncio.sleep(1) 

    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    def mic_callback(indata, frames, time_info, status):
        """Callback invoked by sounddevice stream thread whenever raw data arrives."""
        if status:
            print(status, file=sys.stderr)
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    print(f"🔄 Opening pipeline to WebSocket socket server {WS_URL}...")
    try:
        async with websockets.connect(WS_URL) as websocket:
            
            # Step 2: Receive feedback concurrently
            async def receive_feedback():
                global current_highlighted_idx, scroll_offset
                try:
                    while True:
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        
                        global_line_idx = data.get("global_line_idx", -1)
                        similarity = data.get("similarity", 0.0)
                        
                        # Only update the tracking index if similarity is -1 OR greater than 0.6
                        if global_line_idx != -1 and (similarity == -1 or similarity > 0.6):
                            if global_line_idx != current_highlighted_idx:
                                current_highlighted_idx = global_line_idx
                                scroll_offset = 0 
                        
                        # Scroll 5 lines down if similarity is safely over the 0.6 threshold
                        if similarity > 0.6:
                            max_possible_offset = max(0, len(teleprompter_lines) - current_highlighted_idx - 8)
                            scroll_offset = min(scroll_offset + 5, max_possible_offset)

                        render_ui(
                            latest_transcription=data.get("transcription", ""),
                            similarity=similarity
                        )
                except websockets.exceptions.ConnectionClosed:
                    print("\n🔴 WebSocket processing loop connection closed by backend host.")

            listen_task = asyncio.create_task(receive_feedback())

            # Step 3: Fire up sounddevice capture loop interface 
            block_samples = int(SAMPLE_RATE * BLOCK_DURATION)
            
            mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=block_samples,
                callback=mic_callback
            )

            print("🎙️ MacBook Microphone Active. Speak clearly into your device...")
            with mic_stream:
                while True:
                    audio_block = await audio_queue.get()
                    
                    # Convert float32 matrix to raw PCM 16-bit payload 
                    pcm16_array = (audio_block * 32767.0).astype(np.int16)
                    raw_binary_payload = pcm16_array.tobytes()
                    
                    # Send bytes directly down the pipeline
                    await websocket.send(raw_binary_payload)
                    
    except KeyboardInterrupt:
        print("\n⏹️ Stream intentionally halted by system user sequence.")
    except Exception as e:
        print(f"\n❌ Client runtime execution error: {e}")
    finally:
        listen_task.cancel()
        print("🔌 Cleanup executed. Application closed.")

if __name__ == "__main__":
    try:
        asyncio.run(stream_mic_audio())
    except KeyboardInterrupt:
        pass