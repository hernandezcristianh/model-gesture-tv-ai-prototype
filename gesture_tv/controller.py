"""Máquina de estados para control de TV y transiciones de gestos."""
import time
from .config import (
    CONFIG,
    GESTURE_OPEN_PALM,
    GESTURE_POINTING_UP,
    GESTURE_VICTORY,
    GESTURE_CLOSED_FIST,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
)
from .utils import normalized_distance
from .debouncer import GestureDebouncer

MODE_IDLE = "IDLE"
MODE_ARMED = "ARMED"
MODE_VOLUME = "VOLUME"
MODE_PLAYBACK = "PLAYBACK"


class PinchController:
    def __init__(self, near_thresh, far_thresh, double_tap_window,
                 far_command, close_command, double_command):
        assert near_thresh < far_thresh, "near_thresh debe ser menor que far_thresh"
        self.near_thresh = near_thresh
        self.far_thresh = far_thresh
        self.double_tap_window = double_tap_window
        self.far_command = far_command
        self.close_command = close_command
        self.double_command = double_command
        self.reset()

    def reset(self):
        self.state = "neutral"
        self.pending_close_time = None
        self.pending_command = None

    def update(self, distance, now=None):
        now = time.time() if now is None else now
        command = None

        if distance <= self.near_thresh:
            zone = "close"
        elif distance >= self.far_thresh:
            zone = "far"
        else:
            zone = "neutral"

        if zone != self.state:
            self.state = zone
            if zone == "close":
                if self.pending_close_time is not None and (now - self.pending_close_time) <= self.double_tap_window:
                    self.pending_close_time = None
                    self.pending_command = None
                    command = self.double_command
                else:
                    self.pending_close_time = now
                    self.pending_command = self.close_command
            elif zone == "far":
                self.pending_close_time = None
                self.pending_command = None
                command = self.far_command

        return command

    def poll_pending(self, now=None):
        now = time.time() if now is None else now
        if self.pending_close_time is not None and (now - self.pending_close_time) > self.double_tap_window:
            self.pending_close_time = None
            command, self.pending_command = self.pending_command, None
            return command
        return None


class GestureTVController:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.debouncer = GestureDebouncer(
            persistence=cfg["gesture_persistence"],
            min_confidence=cfg["gesture_min_confidence"],
        )
        self.volume_pinch = PinchController(
            near_thresh=cfg["volume_near_thresh"],
            far_thresh=cfg["volume_far_thresh"],
            double_tap_window=cfg["double_tap_window"],
            far_command="Subir Volumen",
            close_command="Bajar Volumen",
            double_command="Mute",
        )
        self.playback_pinch = PinchController(
            near_thresh=cfg["playback_near_thresh"],
            far_thresh=cfg["playback_far_thresh"],
            double_tap_window=cfg["double_tap_window"],
            far_command="Play",
            close_command="Pausa",
            double_command="Siguiente",
        )
        self.mode = MODE_IDLE
        self.volume_level = cfg["volume_default_level"]
        self.muted = False
        self.playback_state = "Pausa"
        self.next_flash_until = 0.0

    def step(self, gesture_label, gesture_confidence, hand_landmarks, now=None):
        now = time.time() if now is None else now
        events = []
        stable_label, changed = self.debouncer.update(gesture_label, gesture_confidence)

        if changed:
            if stable_label == GESTURE_OPEN_PALM and self.mode == MODE_IDLE:
                self.mode = MODE_ARMED
                events.append(("estado", "Sistema armado (Open_Palm)"))
            elif stable_label == GESTURE_POINTING_UP and self.mode == MODE_ARMED:
                self.mode = MODE_VOLUME
                self.volume_pinch.reset()
                self.playback_pinch.reset()
                events.append(("estado", "Modo volumen (Pointing_Up)"))
            elif stable_label == GESTURE_VICTORY and self.mode == MODE_ARMED:
                self.mode = MODE_PLAYBACK
                self.volume_pinch.reset()
                self.playback_pinch.reset()
                events.append(("estado", "Modo reproduccion (Victory)"))
            elif stable_label == GESTURE_CLOSED_FIST and self.mode != MODE_IDLE:
                self.mode = MODE_IDLE
                self.volume_pinch.reset()
                self.playback_pinch.reset()
                self.volume_level = self.cfg["volume_default_level"]
                self.muted = False
                self.playback_state = "Pausa"
                self.next_flash_until = 0.0
                events.append(("estado", "Detenido (Closed_Fist)"))

        if hand_landmarks is not None:
            if self.mode == MODE_VOLUME:
                dist = normalized_distance(hand_landmarks, THUMB_TIP, INDEX_TIP)
                cmd = self.volume_pinch.update(dist, now) or self.volume_pinch.poll_pending(now)
                if cmd:
                    events.append(("comando", cmd))
            elif self.mode == MODE_PLAYBACK:
                dist = normalized_distance(hand_landmarks, THUMB_TIP, MIDDLE_TIP)
                cmd = self.playback_pinch.update(dist, now) or self.playback_pinch.poll_pending(now)
                if cmd:
                    events.append(("comando", cmd))
        else:
            if self.mode == MODE_VOLUME:
                cmd = self.volume_pinch.poll_pending(now)
                if cmd:
                    events.append(("comando", cmd))
            elif self.mode == MODE_PLAYBACK:
                cmd = self.playback_pinch.poll_pending(now)
                if cmd:
                    events.append(("comando", cmd))

        for kind, value in events:
            if kind == "comando":
                self._apply_command_to_ui(value, now)

        return stable_label, events

    def _apply_command_to_ui(self, command, now):
        step = self.cfg["volume_step"]
        if command == "Subir Volumen":
            self.volume_level = min(100, self.volume_level + step)
            self.muted = False
        elif command == "Bajar Volumen":
            self.volume_level = max(0, self.volume_level - step)
            self.muted = False
        elif command == "Mute":
            self.muted = True
        elif command == "Play":
            self.playback_state = "Play"
        elif command == "Pausa":
            self.playback_state = "Pausa"
        elif command == "Siguiente":
            self.next_flash_until = now + self.cfg["next_flash_seconds"]

    def get_ui_state(self, now=None):
        now = time.time() if now is None else now
        showing_next = now < self.next_flash_until
        return {
            "volume_level": self.volume_level,
            "muted": self.muted,
            "playback_icon": "Siguiente" if showing_next else self.playback_state,
        }