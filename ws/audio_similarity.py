import asyncio
import time
import torch
import numpy as np
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from utils.utils import save_txt
from utils.feat_embed import maximun_similarity

router = APIRouter()


@router.websocket("/ws/audio_similarity")
async def websocket_audio_similarity(websocket: WebSocket):
    """
    Real-time audio similarity websocket endpoint.

    Expected input:
    - Raw PCM16
    - 16kHz
    - Mono
    """

    await websocket.accept()

    print("🟢 WebSocket connected for real-time audio similarity.")

    from main import (
        device,
        torch_dtype,
        processor,
        model_whisper,
        model_bert,
        ls_cluster,
        ls_embed_cluster,
        id_record_section,
        current_metrics,
        TRANSCRIPTION_DIR,
    )

    import main

    SAMPLE_RATE = 16000

    # Keep rolling 6 seconds instead of 10
    MAX_BUFFER_SECONDS = 10
    MAX_BUFFER_BYTES = MAX_BUFFER_SECONDS * SAMPLE_RATE * 2

    # Run inference every 2 seconds
    INFERENCE_INTERVAL = 1.0

    audio_buffer = bytearray()

    last_inference_time = 0

    try:
        while True:

            # ==========================================================
            # RECEIVE AUDIO
            # ==========================================================
            chunk = await websocket.receive_bytes()

            audio_buffer.extend(chunk)

            # Keep only rolling window
            if len(audio_buffer) > MAX_BUFFER_BYTES:
                audio_buffer = audio_buffer[-MAX_BUFFER_BYTES:]

            # ==========================================================
            # THROTTLE INFERENCE
            # ==========================================================
            now = time.time()

            if now - last_inference_time < INFERENCE_INTERVAL:
                continue

            last_inference_time = now

            # ==========================================================
            # VALIDATE CLUSTER INDEX
            # ==========================================================
            local_cur_idx_cluster = main.cur_idx_cluster

            next_idx_cluster = local_cur_idx_cluster + 1

            # IMPORTANT FIX
            if next_idx_cluster + 1 >= len(ls_cluster):

                # await websocket.send_json({
                #     "similarity": 0,
                #     "global_line_idx": -1,
                #     "message": "End of script!"
                # })

                continue

            t_online_start = time.time()

            # ==========================================================
            # AUDIO PREP
            # ==========================================================
            t1 = time.time()

            audio_np = (
                np.frombuffer(audio_buffer, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            input_features = processor(
                audio_np,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).input_features

            input_features = input_features.to(device).to(torch_dtype)

            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english",
                task="transcribe"
            )

            # ==========================================================
            # WHISPER INFERENCE (NON-BLOCKING)
            # ==========================================================
            predicted_ids = await asyncio.to_thread(
                model_whisper.generate,
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )

            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            # ==========================================================
            # SAVE TRANSCRIPT ASYNC
            # ==========================================================
            await asyncio.to_thread(
                save_txt,
                transcription,
                os.path.join(
                    TRANSCRIPTION_DIR,
                    f"{str(id_record_section)}.txt"
                )
            )

            asr_time = time.time() - t1

            # ==========================================================
            # SEMANTIC MATCHING
            # ==========================================================
            t2 = time.time()

            trans_embedding = await asyncio.to_thread(
                model_bert.encode,
                transcription,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            # Similarity search
            max_sim1, max_idx1 = maximun_similarity(
                trans_embedding,
                ls_embed_cluster[next_idx_cluster]
            )
            max_sim2, max_idx2 = maximun_similarity(
                trans_embedding,
                ls_embed_cluster[next_idx_cluster + 1]
            )
            cur_max_sim, cur_max_idx = maximun_similarity(
                trans_embedding,
                ls_embed_cluster[next_idx_cluster - 1]
            )

            # ==========================================================
            # DECISION LOGIC
            # ==========================================================
            max_sim = 0
            global_line_idx = -1

            num_line_per_cluster = 5

            if max_sim1 >= max_sim2:
                max_sim = max_sim1
                best_idx = max_idx1
                chosen_cluster = next_idx_cluster
            else:
                max_sim = max_sim2
                best_idx = max_idx2
                chosen_cluster = next_idx_cluster + 1

            # rollback detection
            if max_sim < cur_max_sim:
                if cur_max_sim < 0.6:
                    max_sim = 0
                else:
                    max_sim = -1
                global_line_idx = (
                    local_cur_idx_cluster * num_line_per_cluster
                ) + cur_max_idx
            else:
                global_line_idx = (
                    chosen_cluster * num_line_per_cluster
                ) + best_idx

            # ==========================================================
            # UPDATE GLOBAL POINTER
            # ==========================================================
            if max_sim > 0.6:
                main.cur_idx_cluster += 1

            semantic_time = time.time() - t2
            online_time = time.time() - t_online_start

            # ==========================================================
            # METRICS
            # ==========================================================
            current_metrics["asr_latency"].append(asr_time)
            current_metrics["semantic_matching"].append(
                semantic_time * 1000
            )
            current_metrics["online_alignment"].append(
                online_time
            )
            current_metrics["cosine_similarity"].append(
                float(max_sim)
            )

            # ==========================================================
            # SEND RESPONSE
            # ==========================================================
            if max_sim > 0.6:
                await websocket.send_json({
                    # "transcription": transcription,
                    "similarity": float(max_sim),
                    "global_line_idx": global_line_idx,
                    "message": "Scrolling!"
                })

    except WebSocketDisconnect:
        print("🔴 WebSocket disconnected.")

    except Exception as e:

        print(f"❌ Error in WebSocket stream: {str(e)}")

        try:
            await websocket.close()
        except:
            pass