"""Funciones geométricas para normalización de distancias de landmarks."""
import numpy as np
from .config import WRIST, MIDDLE_MCP

def get_landmark_xy(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)

def hand_scale(landmarks):
    """Calcula la distancia muñeca - base del dedo medio como unidad de escala."""
    wrist = get_landmark_xy(landmarks, WRIST)
    middle_mcp = get_landmark_xy(landmarks, MIDDLE_MCP)
    return float(np.linalg.norm(middle_mcp - wrist)) + 1e-6

def normalized_distance(landmarks, idx_a, idx_b):
    """Calcula la distancia euclidiana entre dos landmarks invariante a escala."""
    scale = hand_scale(landmarks)
    point_a = get_landmark_xy(landmarks, idx_a)
    point_b = get_landmark_xy(landmarks, idx_b)
    return float(np.linalg.norm(point_a - point_b) / scale)