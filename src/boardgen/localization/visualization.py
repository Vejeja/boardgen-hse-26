from __future__ import annotations

import cv2
import numpy as np

from .geometry import board_corners


def draw_overlay(image, camera, geometries, figures, board_size):
    result = image.copy()
    corners = np.rint(camera.project(board_corners(*board_size))).astype(np.int32)
    cv2.polylines(result, [corners], True, (200, 200, 30), 2, cv2.LINE_AA)
    for item in figures:
        if "placement_xy_m" not in item:
            continue
        geometry = geometries[item["class_id"]]
        color = (50, 190, 70) if item["status"] == "estimated" else (30, 150, 240)
        points = camera.project(geometry.placed(item["placement_xy_m"], item["yaw_deg"]))
        contour = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.polylines(result, [contour], True, color, 1, cv2.LINE_AA)
        theta = np.linspace(0, 2 * np.pi, 64)
        circle = np.column_stack(
            [
                item["xy_m"][0] + item["radius_m"] * np.cos(theta),
                item["xy_m"][1] + item["radius_m"] * np.sin(theta),
                np.zeros(64),
            ]
        )
        cv2.polylines(
            result, [np.rint(camera.project(circle)).astype(np.int32)], True, color, 2, cv2.LINE_AA
        )
        u, v = camera.project([[*item["xy_m"], 0]])[0]
        cv2.drawMarker(result, (round(u), round(v)), color, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(
            result,
            str(item["class_id"]),
            (round(u) + 6, round(v) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def draw_map(template, figures, board_size, max_side=1000):
    width, height = board_size
    scale = max_side / max(width, height)
    result = cv2.resize(template, (max(1, round(width * scale)), max(1, round(height * scale))))
    for item in figures:
        if "xy_m" not in item:
            continue
        x, y = item["xy_m"]
        center = (round((x + width / 2) * scale), round((height / 2 - y) * scale))
        color = (50, 190, 70) if item["status"] == "estimated" else (30, 150, 240)
        cv2.circle(result, center, max(2, round(item["radius_m"] * scale)), color, 2, cv2.LINE_AA)
        cv2.drawMarker(result, center, color, cv2.MARKER_CROSS, 10, 2)
        cv2.putText(
            result,
            str(item["class_id"]),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return result
