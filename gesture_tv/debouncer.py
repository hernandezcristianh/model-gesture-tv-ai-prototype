"""Filtro de persistencia temporal para confirmación de gestos estables."""
from collections import deque
from .config import GESTURE_NONE

class GestureDebouncer:
    def __init__(self, persistence=4, min_confidence=0.6, idle_label=GESTURE_NONE):
        self.persistence = persistence
        self.min_confidence = min_confidence
        self.idle_label = idle_label
        self.window = deque(maxlen=persistence)
        self.stable_label = idle_label

    def update(self, label, confidence):
        if confidence < self.min_confidence:
            label = self.idle_label
        self.window.append(label)

        if len(self.window) == self.window.maxlen and all(x == label for x in self.window):
            new_stable = label
        else:
            new_stable = self.stable_label

        changed = (new_stable != self.stable_label)
        self.stable_label = new_stable
        return new_stable, changed