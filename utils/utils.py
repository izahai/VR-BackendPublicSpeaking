import time
import re

dev_prompt = "## Identity\n" \
"You are a helpful and encouraging English-speaking coach, focusing on fluency, grammar, clarity, structure, and pronunciation.\n\n" \
"## Instructions\n" \
"- Evaluate the text using these categories:\n" \
"  1. Clarity and fluency\n" \
"  2. Grammar and vocabulary\n" \
"  3. Structure and coherence\n" \
"  4. Suggestions for improvement\n" \
"  5. Overall score out of 100\n" \
"- Be constructive, specific, and supportive.\n\n" \
"## Output format\n" \
"Respond using a numbered list for the six evaluation points." \

def save_txt(trans, upload_path):
    with open(upload_path, "a", encoding="utf-8") as f:
        f.write(f"{trans}\n")

def promp_format(trans_path, ori_trans):
    if not trans_path:
        return ""

    with open(trans_path, "r", encoding="utf-8") as f:
        transcription_text = f.read()

    prompt = f"""
        Here is the original transcription on teleprompter:
        \"\"\"
        {ori_trans}
        \"\"\"

        Here is the transcription the user gave:
        \"\"\"
        {transcription_text}
        \"\"\"
        """
    return prompt

def read_transcribed_text(trans_path):
    with open(trans_path, "r", encoding="utf-8") as f:
        return f.read()
    
def read_input_str(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def generate_srt_from_txt(txt_path: str, srt_path: str, segment_duration: int = 10):
    """Convert simple text file lines into timed SRT subtitles."""
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    with open(srt_path, "w") as f:
        for i, line in enumerate(lines):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            start_fmt = time.strftime('%H:%M:%S,000', time.gmtime(start_time))
            end_fmt = time.strftime('%H:%M:%S,000', time.gmtime(end_time))
            f.write(f"{i+1}\n{start_fmt} --> {end_fmt}\n{line}\n\n")

def extract_number(filename):
    match = re.search(r"chunk_(\d+(?:\.\d+)?)", filename)
    return float(match.group(1)) if match else 0