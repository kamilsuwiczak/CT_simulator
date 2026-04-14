import math


def bresenham(x0, y0, x1, y1, width, height):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    if 0 <= x < width and 0 <= y < height:
        points.append((x, y))
    return points


def get_positions(radius, alpha, fan_angle, n_detectors, center_x, center_y):
    emitters = []
    detectors = []

    step = fan_angle / (n_detectors - 1) if n_detectors > 1 else 0
    start_gamma = -fan_angle / 2

    for i in range(n_detectors):
        gamma = start_gamma + i * step

        emitter_x = int(center_x + radius * math.cos(alpha + math.pi - gamma))
        emitter_y = int(center_y + radius * math.sin(alpha + math.pi - gamma))
        emitters.append((emitter_x, emitter_y))

        detector_x = int(center_x + radius * math.cos(alpha + gamma))
        detector_y = int(center_y + radius * math.sin(alpha + gamma))
        detectors.append((detector_x, detector_y))

    return emitters, detectors

