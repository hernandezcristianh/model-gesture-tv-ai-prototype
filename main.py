"""Punto de entrada de la aplicación local de GestureTV."""
import os
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gesture_tv.config import (
    CONFIG,
    GESTURE_NONE,
    HAND_CONNECTIONS,
)
from gesture_tv.tracker import SingleHandTracker
from gesture_tv.controller import GestureTVController, MODE_IDLE
from gesture_tv.ui import draw_volume_bar, draw_playback_icon
from download_model import download_model

MODEL_PATH = "gesture_recognizer.task"

def init_recognizer(model_path):
    if not os.path.exists(model_path):
        download_model(model_path)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    gesture_options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=CONFIG["max_hands"],
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.GestureRecognizer.create_from_options(gesture_options)


def main():
    recognizer = init_recognizer(MODEL_PATH)
    controller = GestureTVController(CONFIG)
    hand_tracker = SingleHandTracker(
        max_jump=CONFIG["hand_track_max_jump"],
        max_missed_seconds=CONFIG["hand_track_max_missed_seconds"],
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        sys.exit(1)

    W, H = CONFIG["frame_width"], CONFIG["frame_height"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    event_log = []
    was_idle = True

    print("--- GestureTV Inicializado ---")
    print("Muestra 'Open_Palm' para armar el sistema.")
    print("Presiona 'q' o ESC en la ventana de video para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Efecto espejo para mayor comodidad visual
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        t0 = time.time()
        result = recognizer.recognize(mp_image)
        now = time.time()
        latency_ms = (now - t0) * 1000.0

        hands = []
        if result.hand_landmarks and result.gestures:
            for lm_list, gest_list in zip(result.hand_landmarks, result.gestures):
                top = gest_list[0]
                hands.append({
                    "label": top.category_name,
                    "confidence": top.score,
                    "landmarks": lm_list,
                })

        chosen = hand_tracker.select(hands, now)

        # Dibujar landmarks
        for h in hands:
            is_chosen = (chosen is not None and h is chosen)
            color = (255, 200, 0) if is_chosen else (120, 120, 120)
            px = [(int(lm.x * W), int(lm.y * H)) for lm in h["landmarks"]]

            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, px[a], px[b], color, 2)
            if is_chosen:
                for x, y in px:
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        hand_landmarks = None
        gesture_label, gesture_confidence = GESTURE_NONE, 0.0
        if chosen is not None:
            hand_landmarks = chosen["landmarks"]
            gesture_label, gesture_confidence = chosen["label"], chosen["confidence"]

        stable_label, events = controller.step(gesture_label, gesture_confidence, hand_landmarks, now)

        for kind, value in events:
            stamp = time.strftime("%H:%M:%S")
            tag = "COMANDO" if kind == "comando" else "ESTADO"
            event_log.append(f"[{stamp}] {tag}: {value}")
        event_log = event_log[-3:]

        # Resetear tracker si entra en reposo
        is_idle = (controller.mode == MODE_IDLE)
        if is_idle and not was_idle:
            hand_tracker.reset()
        was_idle = is_idle

        # Renderizar UI
        if controller.mode != MODE_IDLE:
            ui_state = controller.get_ui_state(now)
            draw_volume_bar(frame, ui_state["volume_level"], ui_state["muted"])
            draw_playback_icon(frame, ui_state["playback_icon"])

        # HUD superior
        hud_text = f"Modo: {controller.mode} | Gesto: {gesture_label} ({gesture_confidence:.2f}) | Latencia: {latency_ms:.0f} ms"
        cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Último evento lanzado
        if event_log:
            cv2.putText(frame, event_log[-1], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("GestureTV - Control Gestual", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()