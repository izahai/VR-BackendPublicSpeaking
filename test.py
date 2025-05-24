import os
from utils.coherence_visual import speed_visulize

TRANSCRIPTION_DIR = "transcriptions"
id_record_folder=13
chart_path = os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_folder)}_chart.png")
speed_visulize(
    os.path.join(TRANSCRIPTION_DIR, f"{str(id_record_folder)}_fb.txt"),
    chart_path,
)