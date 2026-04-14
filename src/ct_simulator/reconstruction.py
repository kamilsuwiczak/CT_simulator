import math

import numpy as np

from .geometry import bresenham, get_positions


def generate_filter(size):
    kernel = np.zeros(size)
    center = size // 2
    for i in range(size):
        if i == center:
            kernel[i] = 1.0
        elif (i - center) % 2 == 0:
            kernel[i] = 0.0
        else:
            kernel[i] = -1.0 / (math.pi ** 2 * (i - center) ** 2)
    return kernel


def apply_filter(sinogram):
    filtered = np.zeros_like(sinogram)
    num_detectors = sinogram.shape[1]
    kernel = generate_filter(num_detectors * 2 + 1)

    for i in range(sinogram.shape[0]):
        row = sinogram[i, :]
        conv_full = np.convolve(row, kernel, mode="full")
        center_start = (len(conv_full) - len(row)) // 2
        filtered[i, :] = conv_full[center_start : center_start + len(row)]

    return filtered


def calculate_rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))


def normalize_array(array):
    array_np = np.asarray(array, dtype=float)
    min_value = float(np.min(array_np))
    max_value = float(np.max(array_np))
    if max_value == min_value:
        return np.zeros_like(array_np, dtype=float)
    return (array_np - min_value) / (max_value - min_value)


def radon_transform(image_array, n_scans, n_detectors, fan_angle_deg, progress_bar=None):
    height, width = image_array.shape
    center_x, center_y = width // 2, height // 2
    radius = math.sqrt(center_x ** 2 + center_y ** 2)
    fan_angle = math.radians(fan_angle_deg)

    sinogram = np.zeros((n_scans, n_detectors))

    for scan in range(n_scans):
        alpha = scan * (math.pi / n_scans)
        emitters, detectors = get_positions(radius, alpha, fan_angle, n_detectors, center_x, center_y)

        for i in range(n_detectors):
            x0, y0 = emitters[i]
            x1, y1 = detectors[i]
            points = bresenham(x0, y0, x1, y1, width, height)
            if points:
                coordinates = np.asarray(points, dtype=np.int32)
                xs = coordinates[:, 0]
                ys = coordinates[:, 1]
                sinogram[scan, i] = float(np.mean(image_array[ys, xs]))
            else:
                sinogram[scan, i] = 0.0

        if progress_bar:
            progress_bar.progress((scan + 1) / (2 * n_scans), text="Generowanie sinogramu...")

    return sinogram


def simulate_tomograph(image_array, n_scans, n_detectors, fan_angle_deg, use_filter=False, progress_bar=None):
    height, width = image_array.shape
    center_x, center_y = width // 2, height // 2
    radius = math.sqrt(center_x ** 2 + center_y ** 2)
    fan_angle = math.radians(fan_angle_deg)

    sinogram = radon_transform(image_array, n_scans, n_detectors, fan_angle_deg, progress_bar)

    display_sinogram = sinogram.copy()

    if use_filter:
        sinogram = apply_filter(sinogram)

    sinogram = normalize_array(sinogram)

    reconstructed = np.zeros((height, width))
    weight_matrix = np.zeros((height, width))

    for scan in range(n_scans):
        alpha = scan * (math.pi / n_scans)
        emitters, detectors = get_positions(radius, alpha, fan_angle, n_detectors, center_x, center_y)

        for i in range(n_detectors):
            x0, y0 = emitters[i]
            x1, y1 = detectors[i]
            points = bresenham(x0, y0, x1, y1, width, height)

            value = sinogram[scan, i]
            for x, y in points:
                reconstructed[y, x] += value
                weight_matrix[y, x] += 1

        if progress_bar:
            progress_bar.progress(0.5 + (scan + 1) / (2 * n_scans), text="Rekonstrukcja obrazu...")

    np.divide(reconstructed, weight_matrix, out=reconstructed, where=weight_matrix != 0)
    reconstructed = normalize_array(reconstructed)

    return display_sinogram, reconstructed

