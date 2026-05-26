import time
import torch
import numpy as np
import librosa
import io
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from utils.utils import save_txt
from utils.feat_embed import maximun_similarity

router = APIRouter()

@router.websocket("/ws/audio_similarity")
async def websocket_audio_similarity(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming. Expects immediate
    raw binary streaming data chunks (PCM 16-bit 16kHz Mono).
    """
    await websocket.accept()
    print("🟢 WebSocket connected for real-time audio similarity.")

    # Access main application state directly
    from main import (
        device, torch_dtype, processor, model_whisper, model_bert,
        ls_cluster, ls_embed_cluster, id_record_section, current_metrics, 
        TRANSCRIPTION_DIR
    )
    import main  # Explicitly import main to handle global state manipulation safely

    SAMPLE_RATE = 16000
    MAX_BUFFER_BYTES = 10 * SAMPLE_RATE * 2 
    
    audio_buffer = bytearray()

    try:
        while True:
            # Receive raw audio chunk bytes directly from the stream
            chunk = await websocket.receive_bytes()
            audio_buffer.extend(chunk)

            # Slide window: Keep only the trailing 10 seconds of audio data
            if len(audio_buffer) > MAX_BUFFER_BYTES:
                audio_buffer = audio_buffer[-MAX_BUFFER_BYTES:]

            # Pull the live, updated cluster index from the main context module
            local_cur_idx_cluster = main.cur_idx_cluster

            # Check cluster bounds relative to server global track point
            next_idx_cluster = local_cur_idx_cluster + 1
            if next_idx_cluster + 1 > len(ls_cluster):
                await websocket.send_json({
                    "similarity": 0,
                    "global_line_idx": -1,
                    "message": "End of script!"
                })
                continue

            t_online_start = time.time()

            # --- Process Audio Buffer ---
            t1 = time.time()
            
            # Convert raw PCM bytes directly to a float32 numpy array
            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Run Whisper Inference
            input_features = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            input_features = input_features.to(device).to(torch_dtype)
            
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = model_whisper.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Save streaming text slice to current section history
            save_txt(transcription, os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}.txt"))
            asr_time = time.time() - t1
            
            # --- Semantic Feature Match ---
            t2 = time.time()
            trans_embedding = model_bert.encode(transcription, convert_to_tensor=True, normalize_embeddings=True)
            
            max_sim1, max_idx1 = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster])
            max_sim2, max_idx2 = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster+1])
            cur_max_sim, cur_max_idx = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster-1])
            
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

            if max_sim < cur_max_sim:
                if cur_max_sim < 0.6:
                    max_sim = 0
                else:
                    max_sim = -1
                global_line_idx = (local_cur_idx_cluster * num_line_per_cluster) + cur_max_idx
            else:
                global_line_idx = (chosen_cluster * num_line_per_cluster) + best_idx
                
            # --- FIXED SCOPE LOGIC ---
            if max_sim > 0.6:
                main.cur_idx_cluster += 1  # Updates the global shared variable cleanly

            semantic_time = time.time() - t2
            online_time = time.time() - t_online_start

            # Append metrics in real-time
            current_metrics["asr_latency"].append(asr_time)
            current_metrics["semantic_matching"].append(semantic_time * 1000)
            current_metrics["online_alignment"].append(online_time)
            current_metrics["cosine_similarity"].append(float(max_sim))

            # Send immediate feedback back through WebSocket frame
            await websocket.send_json({
                "transcription": transcription,
                "similarity": float(max_sim),
                "global_line_idx": global_line_idx,
                "message": "Yes!"
            })
            
    except WebSocketDisconnect:
        print("🔴 WebSocket disconnected.")
    except Exception as e:
        print(f"❌ Error in WebSocket stream: {str(e)}")
        try:
            await websocket.close()
        except:
            pass