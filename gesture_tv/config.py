"""Parámetros de configuración, umbrales y constantes de GestureTV."""

CONFIG = {
    # Reconocimiento de gestos discretos
    "gesture_min_confidence": 0.6,
    "gesture_persistence": 4,

    # Modo volumen (Pointing_Up): pulgar <-> índice
    "volume_near_thresh": 0.35,
    "volume_far_thresh": 1.0,

    # Modo reproducción (Victory): pulgar <-> medio
    "playback_near_thresh": 0.35,
    "playback_far_thresh": 1.0,

    # Ventana de detección de doble pellizco (en segundos)
    "double_tap_window": 0.6,

    # Parámetros de interfaz visual
    "volume_default_level": 50,
    "volume_step": 10,
    "next_flash_seconds": 0.8,

    # Seguimiento unimanual
    "max_hands": 2,
    "hand_track_max_jump": 0.25,
    "hand_track_max_missed_seconds": 1.5,

    # Parámetros de captura
    "frame_width": 640,
    "frame_height": 480,
    "target_fps": 30,
}

# Etiquetas del modelo MediaPipe GestureRecognizer
GESTURE_OPEN_PALM = "Open_Palm"
GESTURE_POINTING_UP = "Pointing_Up"
GESTURE_VICTORY = "Victory"
GESTURE_CLOSED_FIST = "Closed_Fist"
GESTURE_NONE = "None"

# Índices de landmarks clave (convención estándar de MediaPipe)
WRIST = 0
MIDDLE_MCP = 9
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12

# Conexiones del esqueleto de la mano (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),     # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # Meñique
    (0, 17),                                 # Base de la palma
]