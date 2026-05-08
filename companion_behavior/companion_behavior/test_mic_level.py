import queue
import time
import numpy as np
import sounddevice as sd

# Try None first. If wrong, replace with an index from: python -m sounddevice
INPUT_DEVICE = "pulse"
SAMPLE_RATE = 16000

q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print("status:", status)
    q.put(indata.copy())

print("Available devices:")
print(sd.query_devices())

print("\nListening for 10 seconds...")
print("Talk into the MacBook mic. You should see the level increase.\n")

with sd.InputStream(
    device=INPUT_DEVICE,
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=callback,
):
    start = time.time()
    while time.time() - start < 10:
        data = q.get()
        volume = np.linalg.norm(data) * 100
        bar = "#" * min(int(volume), 80)
        print(f"mic level: {volume:8.2f} {bar}")
