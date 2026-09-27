import av
import cv2
import os
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gesture_tv.config import CONFIG, GESTURE_NONE, HAND_CONNECTIONS
from gesture_tv.tracker import SingleHandTracker
from gesture_tv.controller import GestureTVController, MODE_IDLE
from gesture_tv.ui import draw_volume_bar, draw_playback_icon
from download_model import download_model

# Configuración de página
st.set_page_config(page_title="GestureTV Demo", page_icon="📺", layout="centered")
st.title("📺 GestureTV — Control Gestual")
st.caption("Prototipo de visión computacional monocular con MediaPipe")

# Asegurar modelo
MODEL_PATH = "gesture_recognizer.task"
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_PATH)

# Inicializar detector en caché para no recargar por frame
@st.cache_resource
def load_recognizer():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=CONFIG["max_hands"],
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.GestureRecognizer.create_from_options(options)

recognizer = load_recognizer()

# Servidores STUN públicos para conectar WebRTC sin bloqueos de red
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoProcessor:
    def __init__(self):
        self.controller = GestureTVController(CONFIG)
        self.hand_tracker = SingleHandTracker(
            max_jump=CONFIG["hand_track_max_jump"],
            max_missed_seconds=CONFIG["hand_track_max_missed_seconds"],
        )
        self.was_idle = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        H, W, _ = img.shape

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        t0 = time.time()
        result = recognizer.recognize(mp_image)
        now = time.time()
        latency_ms = (now - t0) * 1000.0

        hands = []
        if result.hand_landmarks and result.gestures:
            for lm_list, gest_list in zip(result.hand_landmarks, result.gestures):
                hands.append({
                    "label": gest_list[0].category_name,
                    "confidence": gest_list[0].score,
                    "landmarks": lm_list,
                })

        chosen = self.hand_tracker.select(hands, now)

        # Dibujar landmarks
        for h in hands:
            is_chosen = (chosen is not None and h is chosen)
            color = (255, 200, 0) if is_chosen else (120, 120, 120)
            px = [(int(lm.x * W), int(lm.y * H)) for lm in h["landmarks"]]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, px[a], px[b], color, 2)
            if is_chosen:
                for x, y in px:
                    cv2.circle(img, (x, y), 3, (255, 255, 255), -1)

        hand_landmarks = None
        gesture_label, gesture_confidence = GESTURE_NONE, 0.0
        if chosen is not None:
            hand_landmarks = chosen["landmarks"]
            gesture_label, gesture_confidence = chosen["label"], chosen["confidence"]

        stable_label, events = self.controller.step(gesture_label, gesture_confidence, hand_landmarks, now)

        is_idle = (self.controller.mode == MODE_IDLE)
        if is_idle and not self.was_idle:
            self.hand_tracker.reset()
        self.was_idle = is_idle

        # Dibujar HUD y controles
        if self.controller.mode != MODE_IDLE:
            ui_state = self.controller.get_ui_state(now)
            draw_volume_bar(img, ui_state["volume_level"], ui_state["muted"])
            draw_playback_icon(img, ui_state["playback_icon"])

        hud = f"Modo: {self.controller.mode} | Gesto: {gesture_label} ({gesture_confidence:.2f}) | Latencia: {latency_ms:.0f}ms"
        cv2.putText(img, hud, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Componente de video en vivo
webrtc_streamer(
    key="gesture-tv",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("""
### Instrucciones:
1. Permite el acceso a la cámara y pulsa **START**.
2. Muestra **Open_Palm** (palma abierta) para armar el sistema.
3. Usa **Pointing_Up** para modo volumen o **Victory** para modo reproducción.
4. Cierra el puño (**Closed_Fist**) para apagar o resetear.
""")