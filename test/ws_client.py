# client_stream.py

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
HTTP_URL = "http://127.0.0.1:8000/api/stt_upload"
WS_URL = "ws://127.0.0.1:8000/ws/audio_similarity"

SAMPLE_RATE = 16000
CHANNELS = 1

# IMPORTANT:
# Larger chunk size = more stable
BLOCK_DURATION = 1.0

teleprompter_lines = []
current_highlighted_idx = -1
scroll_offset = 0


# ==========================================================
# FETCH SCRIPT
# ==========================================================
def fetch_teleprompter_script():

    global teleprompter_lines

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

            print(
                f"✅ Loaded {len(teleprompter_lines)} lines."
            )

            return data.get("cur_idx_cluster", 0)

        else:
            print(
                f"⚠️ Failed with status {response.status_code}"
            )

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

    print(f"🗣️ Heard      : {latest_transcription}")
    print(f"🎯 Similarity : {similarity:.4f}")

    print("-" * 70)

    focus_idx = max(0, current_highlighted_idx)

    if current_highlighted_idx == -1:
        start_view = 0
        end_view = min(len(teleprompter_lines), 14)

    else:
        start_view = max(0, focus_idx - 6)
        end_view = min(len(teleprompter_lines), start_view + 14)

    print("📖 SCRIPT WINDOW:\n")

    for idx, line in enumerate(teleprompter_lines):

        if idx < start_view or idx >= end_view:
            continue

        if idx == current_highlighted_idx:
            print(
                f"👉 \033[1;32m[{idx:02d}] {line}\033[0m"
            )
        else:
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

    # ==========================================================
    # MIC CALLBACK
    # ==========================================================
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

                        global_line_idx = data.get(
                            "global_line_idx",
                            -1
                        )

                        similarity = data.get(
                            "similarity",
                            0.0
                        )

                        # update highlight
                        if (
                            global_line_idx != -1
                            and (
                                similarity == -1
                                or similarity > 0.6
                            )
                        ):

                            current_highlighted_idx = (
                                global_line_idx
                            )

                        render_ui(
                            latest_transcription=data.get(
                                "transcription",
                                ""
                            ),
                            similarity=similarity
                        )

                except websockets.exceptions.ConnectionClosed:
                    print(
                        "\n🔴 Connection closed by server."
                    )

            listen_task = asyncio.create_task(
                receive_feedback()
            )

            # ======================================================
            # AUDIO STREAM
            # ======================================================
            block_samples = int(
                SAMPLE_RATE * BLOCK_DURATION
            )

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

                    pcm16_array = (
                        audio_block * 32767.0
                    ).astype(np.int16)

                    raw_binary_payload = (
                        pcm16_array.tobytes()
                    )

                    await websocket.send(
                        raw_binary_payload
                    )

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


# ==========================================================
# ENTRY
# ==========================================================
if __name__ == "__main__":

    try:
        asyncio.run(stream_mic_audio())

    except KeyboardInterrupt:
        pass