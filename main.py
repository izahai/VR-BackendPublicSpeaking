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
from utils.spliter import split_text 
from utils.feat_embed import bert_feat_embed, maximun_similarity
from utils.utils import *
from utils.coherence_visual import speed_visulize
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import base64
import cv2, glob
import ffmpeg
from fastapi.middleware.cors import CORSMiddleware




load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_UPLOAD_DIRECTORY = "record_section"
IMAGE_UPLOAD_DIRECTORY = "img_section"
TRANSCRIPTION_DIR = "transcriptions"
SUBTITLE_DIR = "subtitles"
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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

model_bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

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

# Read input teleprompt script
input_text = read_input_str(os.path.join(BASE_DIR, "input_txt", "input.txt")) 

print("Splitting text into clusters...")
format_txt, ls_cluster, num_lines = split_text(input_text)
print("Extracting features from clusters...")
ls_embed_cluster = bert_feat_embed(model_bert, ls_cluster)
cur_idx_cluster = 0

@app.get("/api/stt_upload")
def ping():
    return {
        "format_txt" : format_txt,
        "cur_idx_cluster" : cur_idx_cluster,
    }

@app.get("/api/GPT_feedback")
async def gpt_feedback():
    try:
        # prompt = promp_format(os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}.txt"), input_text)

        # response = client.chat.completions.create(
        #     model="gpt-4.1",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": dev_prompt
        #         },
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ],
        #     temperature=0.7
        # )

        #feedback = response.choices[0].message.content
        #feedback = feedback.replace("*", "")

        #save_txt(feedback, os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}_fb.txt"))
        feedback = "GPT feedback placeholder..."

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
    id: int = Form(...),
    file: UploadFile = File(...),
    cur_idx_cluster : int = Form(...),
):
    print(f"🟢 Received request: ID={id}, File={file.filename}")
    
    # if not file.filename.endswith(".wav"):
    #     raise HTTPException(status_code=400, detail="File must be a WAV audio file.")
    next_idx_cluster = cur_idx_cluster + 1
    if next_idx_cluster + 1 > len(ls_cluster):
        return {
            "id": id,
            "filename": file.filename,
            "similarity": 0,
            "message": "End of script!"
        }

    upload_record_folder = os.path.join(RECORD_UPLOAD_DIRECTORY, str(id_record_section))
    file_location = os.path.join(upload_record_folder, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    t1 = time.time()
    transcription = pipe(file_location, generate_kwargs={"language": "english"})["text"]
    #transcription = "skibidi skibidi skibidi skibidi skibidi skibidi skibidi skibidi skibidi"
    save_txt(transcription, os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_section)}.txt"))
    t1 = time.time() - t1

    t2 = time.time()
    trans_embedding = model_bert.encode(transcription, convert_to_tensor=True)
    max_sim, max_idx = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster])
    t2 = time.time() - t2

    print(f"Transcription: {transcription}")
    print(f"Best line: {ls_cluster[next_idx_cluster][max_idx]}")
    print(f"Similarity: {max_sim}")
    print(f"Next cluster index: {next_idx_cluster}")
    print(f"Transcription whisper time: {t1:.2f} seconds")
    print(f"Similarity bert time: {t2:.2f} seconds")

    return {
        "id": id,
        "filename": file.filename,
        "transcription": transcription,
        "similarity": max_sim,
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
    print(f"Finalizing video for section {id_record_section}")

    #  --- Collect image frames ---
    image_folder = os.path.join(IMAGE_UPLOAD_DIRECTORY, str(id_record_section))
    images = sorted(glob.glob(f"{image_folder}/*.jpg"))
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    
    #  --- Collect audio segments ---
    audio_folder = os.path.join(RECORD_UPLOAD_DIRECTORY, str(id_record_section))
    audio_files = sorted(glob.glob(f"{audio_folder}/*.wav"), key=extract_number)
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

    # (
    #     ffmpeg
    #     .input(filelist_path, format='concat', safe=0)
    #     .output(
    #         merged_audio_path,
    #         acodec='pcm_s16le',
    #         ar=16000  # (optional) enforce 16 kHz sample rate
    #     )
    #     .run(overwrite_output=True, quiet=False)
    # )

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
    print(f"🎬 Merging audio with video -> {final_output_path}")

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

    return {
        "status": "success",
        "message": "Video merged successfully!",
    }

    # return FileResponse(
    #     final_output_path,
    #     media_type="video/mp4",
    #     filename="output.mp4"
    # )

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