import os
import requests

MODEL_PATH = "gesture_recognizer.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

def download_model(target_path=MODEL_PATH):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"El modelo ya existe en: {target_path} ({os.path.getsize(target_path)} bytes)")
        return target_path

    print(f"Descargando modelo desde {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Modelo descargado exitosamente: {target_path} ({os.path.getsize(target_path)} bytes)")
    return target_path

if __name__ == "__main__":
    download_model()