"""Seguimiento espacial para priorizar y bloquear una sola mano activa."""
import time
import numpy as np
from .utils import get_landmark_xy
from .config import WRIST

class SingleHandTracker:
    def __init__(self, max_jump=0.25, max_missed_seconds=1.5):
        self.max_jump = max_jump
        self.max_missed_seconds = max_missed_seconds
        self.tracked_wrist = None
        self.last_seen_time = None

    def reset(self):
        self.tracked_wrist = None
        self.last_seen_time = None

    def select(self, hands, now=None):
        now = time.time() if now is None else now

        if not hands:
            self._maybe_expire(now)
            return None

        if self.tracked_wrist is None:
            chosen = hands[0]
            self.tracked_wrist = get_landmark_xy(chosen["landmarks"], WRIST)
            self.last_seen_time = now
            return chosen

        distances = [
            float(np.linalg.norm(get_landmark_xy(h["landmarks"], WRIST) - self.tracked_wrist))
            for h in hands
        ]
        best_i = int(np.argmin(distances))

        if distances[best_i] > self.max_jump:
            self._maybe_expire(now)
            return None

        chosen = hands[best_i]
        self.tracked_wrist = get_landmark_xy(chosen["landmarks"], WRIST)
        self.last_seen_time = now
        return chosen

    def _maybe_expire(self, now):
        if self.last_seen_time is not None and (now - self.last_seen_time) > self.max_missed_seconds:
            self.reset()