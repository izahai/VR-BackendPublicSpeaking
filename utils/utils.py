
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

def read_transcribed_text(id_record):
    with open(f"{id_record}.txt", "r", encoding="utf-8") as f:
        return f.read()
    
def read_input_str(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
