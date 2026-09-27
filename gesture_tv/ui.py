"""Funciones de dibujo para overlay de HUD, volumen e iconos de estado."""
import cv2
import numpy as np

def draw_volume_bar(frame, volume_level, muted, x=30, y_top=70, width=42, height=340):
    y_bottom = y_top + height
    white = (255, 255, 255)

    cv2.putText(frame, "VOL", (x - 4, y_top - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y_top), (x + width, y_bottom), white, 2)

    if muted:
        cv2.rectangle(frame, (x + 2, y_top + 2), (x + width - 2, y_bottom - 2), (70, 70, 70), -1)
        cv2.line(frame, (x, y_bottom), (x + width, y_top), (0, 0, 255), 4)
        label = "MUTE"
    else:
        fill_h = int(height * (volume_level / 100.0))
        fill_top = y_bottom - fill_h
        fill_color = (255, 200, 0) if volume_level > 0 else (70, 70, 70)
        if fill_h > 0:
            cv2.rectangle(frame, (x + 2, fill_top), (x + width - 2, y_bottom - 2), fill_color, -1)
        label = f"{volume_level}%"

    cv2.putText(frame, label, (x - 6, y_bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)


def _icon_inner_box(x0, y0, x1, y1, pad_ratio=0.28):
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    return x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y


def _draw_play_glyph(frame, x0, y0, x1, y1, color):
    ix0, iy0, ix1, iy1 = _icon_inner_box(x0, y0, x1, y1)
    pts = np.array([[ix0, iy0], [ix0, iy1], [ix1, (iy0 + iy1) // 2]], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, color)


def _draw_pause_glyph(frame, x0, y0, x1, y1, color):
    ix0, iy0, ix1, iy1 = _icon_inner_box(x0, y0, x1, y1)
    bar_w = max(3, (ix1 - ix0) // 4)
    gap = max(3, (ix1 - ix0) // 4)
    cx = (ix0 + ix1) // 2
    left_x0 = cx - gap // 2 - bar_w
    right_x0 = cx + gap // 2
    cv2.rectangle(frame, (left_x0, iy0), (left_x0 + bar_w, iy1), color, -1)
    cv2.rectangle(frame, (right_x0, iy0), (right_x0 + bar_w, iy1), color, -1)


def _draw_next_glyph(frame, x0, y0, x1, y1, color):
    ix0, iy0, ix1, iy1 = _icon_inner_box(x0, y0, x1, y1)
    mid_x = ix0 + int((ix1 - ix0) * 0.55)
    tri = np.array([[ix0, iy0], [ix0, iy1], [mid_x, (iy0 + iy1) // 2]], dtype=np.int32)
    cv2.fillConvexPoly(frame, tri, color)
    bar_x0 = mid_x + 4
    bar_w = max(3, (ix1 - ix0) // 6)
    cv2.rectangle(frame, (bar_x0, iy0), (min(bar_x0 + bar_w, ix1), iy1), color, -1)


def draw_playback_icon(frame, playback_icon, size=70, right_margin=30, y_top=20):
    W = frame.shape[1]
    x0 = W - right_margin - size
    y0 = y_top
    x1, y1 = x0 + size, y0 + size
    white = (255, 255, 255)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), white, 2)

    if playback_icon == "Play":
        _draw_play_glyph(frame, x0, y0, x1, y1, white)
    elif playback_icon == "Siguiente":
        _draw_next_glyph(frame, x0, y0, x1, y1, white)
    else:
        _draw_pause_glyph(frame, x0, y0, x1, y1, white)