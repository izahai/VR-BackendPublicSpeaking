import asyncio
import json
import sys
import os
import requests
import numpy as np
import sounddevice as sd
import websockets

# ==========================================================
# CONFIG
# ==========================================================
local = False
if local:
    ep = "http://127.0.0.1:8000"
    ws = "ws://127.0.0.1:8000"
else:
    ep = "https://c62d-34-50-185-254.ngrok-free.app"
    ws = "wss://c62d-34-50-185-254.ngrok-free.app"

HTTP_URL = f"{ep}/api/stt_upload"
START_RECORD_URL = f"{ep}/api/start_record"
WS_URL = f"{ws}/ws/audio_similarity"

SAMPLE_RATE = 16000
CHANNELS = 1

BLOCK_DURATION = 1.0

teleprompter_lines = []
current_highlighted_idx = -1

# STRICT SCROLL TRACKER
# Directly maps to the first line index displayed in the visual window
scroll_offset = 0 


# ==========================================================
# INITIALIZE NEW RECORD SESSION
# ==========================================================
def start_new_record_session():
    print(f"🚀 Initializing new session at {START_RECORD_URL}...")
    try:
        response = requests.post(START_RECORD_URL)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Session started successfully!")
            print(f"📁 Section ID: {data.get('id_record_section')}")
            print(f"🧩 Total Clusters: {data.get('numCluster')}")
            return True
        else:
            print(f"⚠️ Failed to start session. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error during session initialization: {e}")
    return False


# ==========================================================
# FETCH SCRIPT
# ==========================================================
def fetch_teleprompter_script():
    global teleprompter_lines, scroll_offset
    print(f"📡 Fetching script from {HTTP_URL}")

    try:
        response = requests.get(HTTP_URL)
        if response.status_code == 200:
            data = response.json()
            raw_txt = data.get("format_txt", "")
            teleprompter_lines = [
                line.strip()
                for line in raw_txt.split("\n")
                if line.strip()
            ]
            print(f"✅ Loaded {len(teleprompter_lines)} lines.")
            
            # Match baseline server position if it already started ahead
            cur_idx_cluster = data.get("cur_idx_cluster", 0)
            scroll_offset = cur_idx_cluster * 5
            return cur_idx_cluster
        else:
            print(f"⚠️ Failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ REST error: {e}")

    return 0


# ==========================================================
# TERMINAL UI
# ==========================================================
def render_ui(latest_transcription, similarity):
    global scroll_offset

    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 70)
    print("🎤 TELEPROMPTER TRACKING")
    print("=" * 70)

    # print(f"🗣️ Heard      : {latest_transcription}")
    print(f"🎯 Similarity : {similarity:.4f}")

    print("-" * 70)

    # ------------------------------------------------------
    # STRICT WINDOW CONTROL
    # Window views exactly 15 lines from our forced scroll offset
    # ------------------------------------------------------
    start_view = max(0, scroll_offset)
    end_view = min(len(teleprompter_lines), start_view + 15)

    print(f"📖 SCRIPT WINDOW (Lines {start_view} to {end_view}):\n")

    # ------------------------------------------------------
    # MULTI-LINE HIGHLIGHT LOGIC (cur+1, cur+2, cur+3)
    # ------------------------------------------------------
    target_highlights = []
    # If no line has been spoken yet, we start highlighting from the very beginning
    base_idx = current_highlighted_idx if current_highlighted_idx != -1 else -1
    
    for offset in [1, 2, 3]:
        t_idx = base_idx + offset
        if 0 <= t_idx < len(teleprompter_lines):
            target_highlights.append(t_idx)

    for idx, line in enumerate(teleprompter_lines):
        if idx < start_view or idx >= end_view:
            continue

        if idx in target_highlights:
            # Highlight upcoming next 3 lines in green
            print(f"👉 \033[1;32m[{idx:02d}] {line}\033[0m")
        elif current_highlighted_idx != -1 and idx <= current_highlighted_idx:
            # MODIFICATION: Show ALL previously completed/spoken lines muted/dimmed gray
            print(f"   \033[2m[{idx:02d}] {line} (Spoken)\033[0m")
        else:
            # Unspoken lines outside of the upcoming green highlights
            print(f"   [{idx:02d}] {line}")

    print("=" * 70)


# ==========================================================
# MAIN STREAM
# ==========================================================
async def stream_mic_audio():
    global current_highlighted_idx
    global scroll_offset

    fetch_teleprompter_script()
    await asyncio.sleep(1)

    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        loop.call_soon_threadsafe(
            audio_queue.put_nowait,
            indata.copy()
        )

    print(f"🔄 Connecting to {WS_URL}")

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=120,
            close_timeout=10,
            max_size=10_000_000
        ) as websocket:

            # ======================================================
            # RECEIVE LOOP
            # ======================================================
            async def receive_feedback():
                global current_highlighted_idx
                global scroll_offset

                try:
                    while True:
                        msg = await websocket.recv()
                        data = json.loads(msg)

                        global_line_idx = data.get("global_line_idx", -1)
                        similarity = data.get("similarity", 0.0)

                        # Check validation criteria
                        is_valid_match = global_line_idx != -1 and (similarity == -1 or similarity > 0.6)
                        
                        # MODIFICATION: Only advance if the index is strictly greater than the previous one
                        is_moving_forward = global_line_idx > current_highlighted_idx

                        if is_valid_match and is_moving_forward:
                            current_highlighted_idx = global_line_idx

                            # --------------------------------------------------
                            # STRICT SERVER-DRIVEN SCROLL LOGIC
                            # Moved inside the 'is_moving_forward' block so scrolling
                            # only triggers when the text actually progresses.
                            # --------------------------------------------------
                            if similarity > 0.6:
                                scroll_offset += 5

                        # Render UI on every message to update "Heard" and "Similarity" stats 
                        # even if the highlighted line index didn't advance.
                        render_ui(
                            latest_transcription=data.get("transcription", ""),
                            similarity=similarity
                        )

                except websockets.exceptions.ConnectionClosed:
                    print("\n🔴 Connection closed by server.")

            listen_task = asyncio.create_task(receive_feedback())

            # ======================================================
            # AUDIO STREAM
            # ======================================================
            block_samples = int(SAMPLE_RATE * BLOCK_DURATION)

            mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=block_samples,
                callback=mic_callback
            )

            print("🎙️ Microphone active...")

            with mic_stream:
                while True:
                    audio_block = await audio_queue.get()
                    pcm16_array = (audio_block * 32767.0).astype(np.int16)
                    raw_binary_payload = pcm16_array.tobytes()
                    await websocket.send(raw_binary_payload)

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user.")
    except Exception as e:
        print(f"\n❌ Runtime error: {e}")
    finally:
        try:
            listen_task.cancel()
        except:
            pass
        print("🔌 Cleanup complete.")


if __name__ == "__main__":
    session_ok = start_new_record_session()
    
    if session_ok:
        try:
            asyncio.run(stream_mic_audio())
        except KeyboardInterrupt:
            pass
    else:
        print("🚨 Aborting startup because recording folder initialization failed.")