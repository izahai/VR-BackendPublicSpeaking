import whisper
import time
# import os
# import certifi

# os.environ["SSL_CERT_FILE"] = certifi.where()


# Load Whisper model on Metal (Apple Silicon acceleration)
# model = whisper.load_model("large-v3-turbo")
model = whisper.load_model("tiny")


t1 = time.time()
result = model.transcribe("sample.mp3")
print("Model transcribe time:", time.time() - t1)

print(result["text"])


