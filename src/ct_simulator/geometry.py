import math

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


@njit
def _bresenham_numba(x0, y0, x1, y1, width, height):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    max_len = dx if dx > dy else dy
    points = np.empty((max_len + 1, 2), dtype=np.int64)
    count = 0
    x = x0
    y = y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if 0 <= x < width and 0 <= y < height:
                points[count, 0] = x
                points[count, 1] = y
                count += 1
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if 0 <= x < width and 0 <= y < height:
                points[count, 0] = x
                points[count, 1] = y
                count += 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    if 0 <= x < width and 0 <= y < height:
        points[count, 0] = x
        points[count, 1] = y
        count += 1

    return points[:count]


@njit
def _get_positions_numba(radius, alpha, fan_angle, n_detectors, center_x, center_y):
    emitters = np.empty((n_detectors, 2), dtype=np.int64)
    detectors = np.empty((n_detectors, 2), dtype=np.int64)

    step = fan_angle / (n_detectors - 1) if n_detectors > 1 else 0.0
    start_gamma = -fan_angle / 2.0

    for i in range(n_detectors):
        gamma = start_gamma + i * step

        emitter_x = int(center_x + radius * math.cos(alpha + math.pi - gamma))
        emitter_y = int(center_y + radius * math.sin(alpha + math.pi - gamma))
        emitters[i, 0] = emitter_x
        emitters[i, 1] = emitter_y

        detector_x = int(center_x + radius * math.cos(alpha + gamma))
        detector_y = int(center_y + radius * math.sin(alpha + gamma))
        detectors[i, 0] = detector_x
        detectors[i, 1] = detector_y

    return emitters, detectors


def bresenham(x0, y0, x1, y1, width, height):
    points = _bresenham_numba(x0, y0, x1, y1, width, height)
    return [(int(point[0]), int(point[1])) for point in points]


def get_positions(radius, alpha, fan_angle, n_detectors, center_x, center_y):
    emitters, detectors = _get_positions_numba(radius, alpha, fan_angle, n_detectors, center_x, center_y)
    return [(int(point[0]), int(point[1])) for point in emitters], [
        (int(point[0]), int(point[1])) for point in detectors
    ]
