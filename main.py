from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import shutil
import whisper
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
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIRECTORY = "videos"
TRANSCRIPTION_DIRECTORY = "transcriptions"
os.makedirs(TRANSCRIPTION_DIRECTORY, exist_ok=True)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


app = FastAPI()
client = OpenAI()

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

id_record_folder = len([
    name for name in os.listdir(UPLOAD_DIRECTORY)
    if os.path.isdir(os.path.join(UPLOAD_DIRECTORY, name))
])

# Read input teleprompt script
input_text = read_input_str(os.path.join(BASE_DIR, "input_txt", "input.txt")) 

print("Splitting text into clusters...")
format_txt, ls_cluster, num_lines = split_text(input_text)
print("Extracting features from clusters...")
ls_embed_cluster = bert_feat_embed(model_bert, ls_cluster)
cur_idx_cluster = 0

UPLOAD_DIRECTORY = "videos"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.get("/api/stt_uploads")
def ping():
    return {
        "format_txt" : format_txt,
        "cur_idx_cluster" : cur_idx_cluster,
    }

@app.get("/api/GPT_feedback")
def gpt_feedback():
    try:
        prompt = promp_format(os.path.join(TRANSCRIPTION_DIRECTORY, f"{str(id_record_folder)}.txt"), input_text)

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
        save_txt(feedback, os.path.join(TRANSCRIPTION_DIRECTORY, f"{str(id_record_folder)}_fb.txt"))
        feedback += "\n" + read_transcribed_text(id_record_folder)
        return {"feedback": feedback}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/start_record")
def create_new_record_folder():
    global id_record_folder
    id_record_folder += 1
    new_folder_path = os.path.join(UPLOAD_DIRECTORY, str(id_record_folder))
    
    os.makedirs(new_folder_path, exist_ok=True)

    return {
        "new_id_folder": id_record_folder,
        "numCluster": len(ls_cluster),
        "message": "Create successfully!",
    }

@app.post("/api/stt_uploads")
def upload_video(
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

    upload_record_folder = os.path.join(UPLOAD_DIRECTORY, str(id_record_folder))
    file_location = os.path.join(upload_record_folder, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    t1 = time.time()
    #transcription = model_whisper.transcribe(file_location, language="en")["text"]
    transcription = pipe(file_location)["text"]
    save_txt(transcription, os.path.join(TRANSCRIPTION_DIRECTORY, f"{str(id_record_folder)}.txt"))
    #transcription = "Skibidi skibidi skibidi !!!"
    t1 = time.time() - t1

    t2 = time.time()
    trans_embedding = model_bert.encode(transcription, convert_to_tensor=True)
    max_sim, max_idx = maximun_similarity(trans_embedding, ls_embed_cluster[next_idx_cluster])
    t2 = time.time() - t2

    print(f"🗣️ Transcription: {transcription}")
    print(f"🔍 Best line: {ls_cluster[next_idx_cluster][max_idx]}")
    print(f"🔍 Similarity: {max_sim}")
    print(f"🔍 Next cluster index: {next_idx_cluster}")
    print(f"⏱️ Transcription whisper time: {t1:.2f} seconds")
    print(f"⏱️ Similarity bert time: {t2:.2f} seconds")

    return {
        "id": id,
        "filename": file.filename,
        "transcription": transcription,
        "similarity": max_sim,
        "message": "Yes!"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)