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


def _normalize_01(image):
    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:
        return np.zeros_like(image)
    return np.interp(image, (min_val, max_val), (0, 1))


def simulate_tomograph(
    image_array,
    n_scans,
    n_detectors,
    fan_angle_deg,
    use_filter=False,
    progress_bar=None,
    collect_rmse_per_iteration=False,
):
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
            ray_sum = sum(image_array[y, x] for x, y in points)
            sinogram[scan, i] = ray_sum / len(points) if points else 0

        if progress_bar:
            progress_bar.progress((scan + 1) / (2 * n_scans), text="Generowanie sinogramu...")

    display_sinogram = sinogram.copy()

    if use_filter:
        sinogram = apply_filter(sinogram)

    sinogram = np.interp(sinogram, (sinogram.min(), sinogram.max()), (0, 1))

    reconstructed = np.zeros((height, width))
    weight_matrix = np.zeros((height, width))
    rmse_per_iteration = [] if collect_rmse_per_iteration else None

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

        if collect_rmse_per_iteration:
            current = np.zeros_like(reconstructed)
            np.divide(reconstructed, weight_matrix, out=current, where=weight_matrix != 0)
            current = _normalize_01(current)
            rmse_per_iteration.append(calculate_rmse(image_array, current))

        if progress_bar:
            progress_bar.progress(0.5 + (scan + 1) / (2 * n_scans), text="Rekonstrukcja obrazu...")

    np.divide(reconstructed, weight_matrix, out=reconstructed, where=weight_matrix != 0)
    reconstructed = _normalize_01(reconstructed)

    if collect_rmse_per_iteration:
        return display_sinogram, reconstructed, rmse_per_iteration
    return display_sinogram, reconstructed


def analyze_rmse_statistics(image_array, n_scans, n_detectors, fan_angle_deg):
    _, _, rmse_per_iteration = simulate_tomograph(
        image_array,
        n_scans,
        n_detectors,
        fan_angle_deg,
        use_filter=False,
        progress_bar=None,
        collect_rmse_per_iteration=True,
    )

    # Rosnace poziomy dokladnosci probkowania dla trzech parametrow modelu.
    n_scans_levels = sorted({max(10, n_scans // 2), n_scans, min(720, int(n_scans * 1.5))})
    n_detectors_levels = sorted({max(10, n_detectors // 2), n_detectors, min(720, int(n_detectors * 1.5))})
    fan_angle_levels = sorted({max(45, fan_angle_deg // 2), fan_angle_deg, min(360, int(fan_angle_deg * 1.5))})

    rmse_vs_scans = []
    for scans in n_scans_levels:
        _, reconstructed = simulate_tomograph(
            image_array, scans, n_detectors, fan_angle_deg, use_filter=False, progress_bar=None
        )
        rmse_vs_scans.append(calculate_rmse(image_array, reconstructed))

    rmse_vs_detectors = []
    for detectors in n_detectors_levels:
        _, reconstructed = simulate_tomograph(
            image_array, n_scans, detectors, fan_angle_deg, use_filter=False, progress_bar=None
        )
        rmse_vs_detectors.append(calculate_rmse(image_array, reconstructed))

    rmse_vs_fan_angle = []
    for fan in fan_angle_levels:
        _, reconstructed = simulate_tomograph(
            image_array, n_scans, n_detectors, fan, use_filter=False, progress_bar=None
        )
        rmse_vs_fan_angle.append(calculate_rmse(image_array, reconstructed))

    _, reconstructed_no_filter = simulate_tomograph(
        image_array, n_scans, n_detectors, fan_angle_deg, use_filter=False, progress_bar=None
    )
    _, reconstructed_with_filter = simulate_tomograph(
        image_array, n_scans, n_detectors, fan_angle_deg, use_filter=True, progress_bar=None
    )

    return {
        "rmse_per_iteration": rmse_per_iteration,
        "sampling": {
            "n_scans": {"values": n_scans_levels, "rmse": rmse_vs_scans},
            "n_detectors": {"values": n_detectors_levels, "rmse": rmse_vs_detectors},
            "fan_angle_deg": {"values": fan_angle_levels, "rmse": rmse_vs_fan_angle},
        },
        "filter_comparison": {
            "without_filter": calculate_rmse(image_array, reconstructed_no_filter),
            "with_filter": calculate_rmse(image_array, reconstructed_with_filter),
        },
    }

