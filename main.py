from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import shutil
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
import time
import torch
import uvicorn
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import base64
import cv2, glob
import ffmpeg
from statistics import mean, stdev
import json
from fastapi.middleware.cors import CORSMiddleware
import librosa

from ws.audio_similarity import router as ws_similarity_router

from utils.spliter import split_text 
from utils.feat_embed import bert_feat_embed, maximun_similarity
from utils.utils import *
from utils.coherence_visual import speed_visulize




load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_UPLOAD_DIRECTORY = "record_section"
IMAGE_UPLOAD_DIRECTORY = "img_section"
TRANSCRIPTION_DIR = "transcriptions"
SUBTITLE_DIR = "subtitles"
METRICS_DIR = "metrics"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
os.makedirs(RECORD_UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(SUBTITLE_DIR, exist_ok=True)


app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your Quest device IP/domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_similarity_router)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

text_encoder = "BAAI/bge-large-en-v1.5"
# text_encoder = "sentence-transformers/all-mpnet-base-v2"
model_bert = SentenceTransformer(text_encoder)

model_bert.to(device)
model_whisper.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_whisper,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

id_record_section = len([
    name for name in os.listdir(RECORD_UPLOAD_DIRECTORY)
    if os.path.isdir(os.path.join(RECORD_UPLOAD_DIRECTORY, name))
])

current_metrics = {
    "asr_latency": [],
    "semantic_matching": [],
    "online_alignment": [],
    "cosine_similarity": [],
    "video_render": None,
}

# Read input teleprompt script
input_text = read_input_str(os.path.join(BASE_DIR, "input_txt", "input.txt")) 

print("Splitting text into clusters...")
format_txt, ls_cluster, num_lines = split_text(input_text)
print("Extracting features from clusters...")
offline_encoding_time = time.time()
ls_embed_cluster = bert_feat_embed(model_bert, ls_cluster)
offline_encoding_time = time.time() - offline_encoding_time
OFFLINE_ENCODING_TIME = offline_encoding_time
cur_idx_cluster = 0

def save_metrics(section_id: int):
    def mean_std(values):
        if len(values) < 2:
            return {
                "mean": values[0] if values else 0.0,
                "std": 0.0
            }
        return {
            "mean": mean(values),
            "std": stdev(values)
        }

    metrics_summary = {
        "section_id": section_id,
        "asr_latency_s": mean_std(current_metrics["asr_latency"]),
        "semantic_matching_ms": mean_std(current_metrics["semantic_matching"]),
        "online_speech_alignment_s": mean_std(current_metrics["online_alignment"]),
        "average_cosine_similarity": mean_std(current_metrics["cosine_similarity"]),
        "offline_semantic_encoding_s": OFFLINE_ENCODING_TIME,
        "video_render_s": current_metrics["video_render"],
    }

    path = os.path.join(METRICS_DIR, f"section_{section_id}.json")
    with open(path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    return metrics_summary


@app.get("/api/stt_upload")
def ping():
    return {
        "teleprompter_script" : format_txt,
    }

@app.get("/api/GPT_feedback")
async def gpt_feedback():
    try:
        prompt = promp_format(os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}.txt"), input_text)

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": dev_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        feedback = response.choices[0].message.content
        feedback = feedback.replace("*", "")

        save_txt(feedback, os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}_fb.txt"))
        # feedback = "GPT feedback placeholder..."

        chart_path = os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}_chart.png")
        transcribed_path = os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}.txt")
        
        feedback += "\n\n\n" + "Transcribed text:\n" + read_transcribed_text(transcribed_path)

        # Create speed line chart
        speed_visulize(
            transcribed_path,
            chart_path,
        )
        with open(chart_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")


        return {
            "feedback": feedback,
            "img_base64": img_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/start_record")
def create_new_record_folder():
    global id_record_section
    global cur_idx_cluster
    
    cur_idx_cluster = 0
    id_record_section += 1
    audio_fpath = os.path.join(RECORD_UPLOAD_DIRECTORY, str(id_record_section))
    img_fpath = os.path.join(IMAGE_UPLOAD_DIRECTORY, str(id_record_section))
    
    os.makedirs(audio_fpath, exist_ok=True)
    os.makedirs(img_fpath, exist_ok=True)

    return {
        "id_record_section": id_record_section,
        "teleprompter_script": format_txt,
        "numCluster": len(ls_cluster),
        "message": "Create successfully!",
    }

@app.post("/api/stt_upload")
def upload_audio_record(
    file: UploadFile = File(...),
):
    print(f"🟢 Received request: ID={id}, File={file.filename}")
    
    next_idx_cluster = cur_idx_cluster + 1
    if next_idx_cluster + 1 > len(ls_cluster):
        return {
            "similarity": 0,
            "global_line_idx": -1,  # Added fallback value for end of script
            "message": "End of script!"
        }

    upload_record_folder = os.path.join(RECORD_UPLOAD_DIRECTORY, str(id_record_section))
    file_location = os.path.join(upload_record_folder, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    t_online_start = time.time()

    t1 = time.time()
    audio_array, sampling_rate = librosa.load(file_location, sr=16000)
    
    # Process and generate
    input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
    input_features = input_features.to(device).to(torch_dtype)
    
    # Force language generation targets
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    
    predicted_ids = model_whisper.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    save_txt(transcription, os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}.txt"))
    asr_time = time.time() - t1
    
    t2 = time.time()
    trans_embedding = model_bert.encode(transcription, convert_to_tensor=True, normalize_embeddings=True)
    
    max_sim1, max_idx1 = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster])
    max_sim2, max_idx2 = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster+1])
    cur_max_sim, cur_max_idx = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster-1])
    
    max_sim = 0
    global_line_idx = -1
    num_line_per_cluster = 5  # Ensure this matches your script's partitioning logic
    
    # --- Choose the max similarity cluster (next or 2 steps next) ---
    if max_sim1 >= max_sim2:
        max_sim = max_sim1
        best_idx = max_idx1
        chosen_cluster = next_idx_cluster
    else:
        max_sim = max_sim2
        best_idx = max_idx2
        chosen_cluster = next_idx_cluster + 1

    # --- Do not scroll if the next cluster don't exceed cur cluster ---
    if max_sim < cur_max_sim:
        max_sim = -1
        global_line_idx = (cur_idx_cluster * num_line_per_cluster) + cur_max_idx
        best_line_text = "Max similarity is still in the current cluster"
    else:
        # Calculate the absolute line index across the entire script
        global_line_idx = (chosen_cluster * num_line_per_cluster) + best_idx
        best_line_text = ls_cluster[chosen_cluster][best_idx]

    semantic_time = time.time() - t2
    online_time = time.time() - t_online_start

    print(f"Transcription: {transcription}")
    print(f"Best line: {best_line_text}")
    print(f"Similarity: {max_sim}")
    print(f"Global Line Index: {global_line_idx}")
    print(f"Next cluster index: {next_idx_cluster}")
    
    print(f"Transcription whisper time: {asr_time:.2f} seconds")
    print(f"Similarity bert time: {semantic_time:.2f} seconds")
    print(f"Total online alignment time: {online_time:.2f} seconds")

    current_metrics["asr_latency"].append(asr_time)
    current_metrics["semantic_matching"].append(semantic_time * 1000)  # ms
    current_metrics["online_alignment"].append(online_time)
    current_metrics["cosine_similarity"].append(float(max_sim))

    return {
        "id": id,
        "filename": file.filename,
        "transcription": transcription,
        "similarity": max_sim,
        "global_line_idx": global_line_idx,  # Explicitly included in JSON response
        "message": "Yes!"
    }

@app.post("/api/image_upload")
async def upload_image(
    file: UploadFile = File(...)
):
    img_fpath = os.path.join(IMAGE_UPLOAD_DIRECTORY, str(id_record_section))
    if not os.path.exists(img_fpath):
        return {"error": "Image fpath does not exist"}
    
    filepath =  os.path.join(img_fpath, file.filename)    
    with open(filepath, "wb") as f:
        f.write(await file.read())

    return {"status": "ok", "file": file.filename}
    
@app.get("/api/finalize_video")
async def finalize_video():
    video_start_time = time.time()
    print(f"Finalizing video for section {id_record_section}")

    #  --- Collect image frames ---
    image_folder = os.path.join(IMAGE_UPLOAD_DIRECTORY, str(id_record_section))
    images = sorted(glob.glob(f"{image_folder}/*.jpg"))
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    
    #  --- Collect audio segments ---
    audio_folder = os.path.join(RECORD_UPLOAD_DIRECTORY, str(id_record_section))
    audio_files = [
        f for f in sorted(glob.glob(f"{audio_folder}/*.wav"), key=extract_number)
        if not os.path.basename(f).startswith("merged_")
    ]

    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio files uploaded")
    
    # --- Create base video from images ---
    frame = cv2.imread(images[0])
    h, w, _ = frame.shape
    backend_video_name = f"output_{id_record_section}.mp4"
    backend_video_path = os.path.join(image_folder, backend_video_name)
    fps = 8

    print(f"Creating base video at {backend_video_path}")

    out = cv2.VideoWriter(backend_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()

    # --- Generate subtitle from transcription ---
    transcribed_path = os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}.txt")
    srt_path = os.path.join(SUBTITLE_DIR, f"{str(id_record_section)}.srt")
    generate_srt_from_txt(transcribed_path, srt_path, segment_duration=10)

    # --- Concatenate audio segments using ffmpeg-python ---
    filelist_path = os.path.join(audio_folder, "filelist.txt")
    with open(filelist_path, "w") as f:
        for a in audio_files:
            f.write(f"file '{os.path.abspath(a)}'\n")

    merged_audio_path = os.path.join(audio_folder, "merged_audio.wav")
    print(f"Concatenating {len(audio_files)} audio clips...")

    (
        ffmpeg
        .input(filelist_path, format='concat', safe=0)
        .output(
            merged_audio_path,
            acodec='pcm_s16le',
            ar=16000,
            vsync='cfr'
        )
        .global_args('-fflags', '+genpts')
        .global_args('-async', '1')
        .run(overwrite_output=True, quiet=False)
    )


    # --- Merge video + audio ---
    final_output_path = os.path.join(image_folder, f"final_{id_record_section}.mp4")
    print(f"Merging audio with video -> {final_output_path}")

    video_in = ffmpeg.input(backend_video_path)
    audio_in = ffmpeg.input(merged_audio_path)

    (
        ffmpeg
        .output(
            video_in,
            audio_in,
            final_output_path,
            vf=f"subtitles={srt_path}:force_style='Fontsize=24,PrimaryColour=&HFFFFFF&'",
            af="volume=2.0", # boost audio volume x2
            vcodec='libx264',  
            acodec='aac',      # encode audio to AAC for MP4 container
            movflags='+faststart', # for streaming
            shortest=None      # stop at the shortest stream
        )
        .run(overwrite_output=True, quiet=False)
    )

    print(f"(v) Final video generated at {final_output_path}")

    current_metrics["video_render"] = time.time() - video_start_time

    metrics = save_metrics(id_record_section)

    return {
        "status": "success",
        "message": "Video merged successfully!",
    }

@app.post("/api/image_zip_upload")
async def upload_image_zip(file: UploadFile = File(...)):
    import zipfile, io, os
    img_dir = os.path.join(IMAGE_UPLOAD_DIRECTORY, str(id_record_section))
    os.makedirs(img_dir, exist_ok=True)

    content = await file.read()
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(img_dir)

    return {"status": "ok", "num_files": len(zf.namelist())}

@app.get("/api/stream_video")
async def stream_video(request: Request):
    """
    Stream the generated MP4 to clients (supports HTTP Range for partial loading).
    This allows Unity VideoPlayer (or browsers) to start playback immediately.
    """
    video_path = os.path.join(
        IMAGE_UPLOAD_DIRECTORY,
        str(id_record_section),
        f"final_{id_record_section}.mp4"
    )

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("range")
    chunk_size = 1024 * 1024  # 1 MB chunks

    def iterfile(start: int = 0, end: int = None):
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = (end or file_size) - start
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if range_header:
        # Example: "bytes=1000-"
        bytes_range = range_header.replace("bytes=", "").split("-")
        start = int(bytes_range[0]) if bytes_range[0] else 0
        end = int(bytes_range[1]) if len(bytes_range) > 1 and bytes_range[1] else file_size - 1
        length = end - start + 1

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(
            iterfile(start, end + 1),
            status_code=206,
            headers={
                **headers,
                "Cache-Control": "no-store",
            },
        )


    # No Range header → send full file
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
    }

    return StreamingResponse(iterfile(), headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)