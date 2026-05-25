import nest_asyncio, threading, time, requests, socket
from pyngrok import ngrok
import uvicorn
import os

# Patch asyncio loop for Colab
nest_asyncio.apply()

# Fake OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "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"
NGROK_TOKEN = "2wnEVIZZXhQL1RF0i4LKzRWk4KK_7aKwWMr2mv1r8VWghNJ3H"
PORT = 8000

# 1) Start Uvicorn in a background thread
def run_app():
    # IMPORTANT: Ensure your FastAPI app instance is defined as `app` inside a file named `main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

server_thread = threading.Thread(target=run_app, daemon=True)
server_thread.start()

# 2) Poll until the server is actually ready
def wait_for_ready(url, timeout=180):
    t0 = time.time()
    while True:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code in (200, 404):  # docs or not found both mean server is responding
                return True
        except Exception:
            pass
        if time.time() - t0 > timeout:
            raise RuntimeError("Uvicorn didn't become ready in time.")
        time.sleep(2)

local_url = f"http://127.0.0.1:{PORT}/docs"
print("⏳ Waiting for server to be ready at", local_url)
wait_for_ready(local_url)
print("✅ Server is up at", local_url)

# 3) Open ngrok tunnel only after server is up
ngrok.set_auth_token(NGROK_TOKEN)
public_tunnel = ngrok.connect(PORT, "http")
print("🌐 Public URL:", public_tunnel.public_url)

# --- NEW ADDITION BELOW: Keep main thread alive to stream logs ---
print("\n🚀 Server is running and streaming logs below. Press CTRL+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 Shutting down ngrok tunnel and server...")
    try:
        ngrok.disconnect(public_tunnel.public_url)
    except Exception:
        pass