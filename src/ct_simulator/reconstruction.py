import math
import numpy as np
from numba import njit

from .geometry import _bresenham_numba, _get_positions_numba

NUMBA_AVAILABLE = True


@njit
def generate_filter(size):
    kernel = np.zeros(size)
    center = size // 2
    for i in range(size):
        k = i - center
        if k == 0:
            kernel[i] = 1.0
        elif k % 2 == 0:
            kernel[i] = 0.0
        else:
            kernel[i] = -4.0 / (np.pi ** 2 * k ** 2)
    return kernel


def apply_filter(sinogram):
    filtered = np.zeros_like(sinogram)
    num_detectors = sinogram.shape[1]
    kernel = generate_filter(num_detectors * 2 + 1)

    for i in range(sinogram.shape[0]):
        row = sinogram[i, :]
        conv_full = np.convolve(row, kernel, mode="full")
        center_start = (len(conv_full) - len(row)) // 2
        filtered[i, :] = conv_full[center_start: center_start + len(row)]

    return filtered


# 3. ODPORNA NORMALIZACJA (dla obrazu zrekonstruowanego)
def normalize_robust(image):
    """
    Normalizacja ignorująca ekstremalne wartości (artefakty filtra),
    które psują kontrast i zawyżają błąd RMSE.
    """
    img_flat = image.flatten()

    p_low = np.percentile(img_flat, 2)
    p_high = np.percentile(img_flat, 98)

    clipped = np.clip(image, p_low, p_high)

    if p_high == p_low:
        return np.zeros_like(clipped)

    return (clipped - p_low) / (p_high - p_low)


@njit
def _normalize_01_numba(image):
    flat = image.ravel()
    min_val = flat[0]
    max_val = flat[0]

    for i in range(flat.size):
        value = flat[i]
        if value < min_val:
            min_val = value
        elif value > max_val:
            max_val = value

    normalized = np.zeros_like(image)
    if max_val == min_val:
        return normalized

    scale = 1.0 / (max_val - min_val)
    normalized_flat = normalized.ravel()
    for i in range(flat.size):
        normalized_flat[i] = (flat[i] - min_val) * scale
    return normalized


@njit
def calculate_rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))


def _normalize_01(image):
    return _normalize_01_numba(np.asarray(image, dtype=np.float64))


def normalize_array(image):
    return _normalize_01(image)


@njit
def _ray_mean_numba(image_array, x0, y0, x1, y1, width, height):
    points = _bresenham_numba(x0, y0, x1, y1, width, height)
    total = 0.0
    count = points.shape[0]

    if count == 0:
        return 0.0

    for i in range(count):
        total += image_array[points[i, 1], points[i, 0]]

    return total / count


def radon_transform(image_array, n_scans, n_detectors, fan_angle_deg, progress_bar=None):
    height, width = image_array.shape
    center_x, center_y = width // 2, height // 2
    radius = math.sqrt(center_x ** 2 + center_y ** 2)
    fan_angle = math.radians(fan_angle_deg)

    sinogram = np.zeros((n_scans, n_detectors))

    for scan in range(n_scans):
        alpha = scan * (math.pi / n_scans)
        emitters, detectors = _get_positions_numba(radius, alpha, fan_angle, n_detectors, center_x, center_y)

        for i in range(n_detectors):
            x0, y0 = emitters[i]
            x1, y1 = detectors[i]
            sinogram[scan, i] = _ray_mean_numba(image_array, x0, y0, x1, y1, width, height)

        if progress_bar:
            progress_bar.progress((scan + 1) / (2 * n_scans), text="Generowanie sinogramu...")

    return sinogram


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

    sinogram = radon_transform(image_array, n_scans, n_detectors, fan_angle_deg, progress_bar)

    display_sinogram = sinogram.copy()

    if use_filter:
        sinogram = apply_filter(sinogram)

    sinogram = normalize_array(sinogram)

    reconstructed = np.zeros((height, width))
    weight_matrix = np.zeros((height, width))
    rmse_per_iteration = [] if collect_rmse_per_iteration else None

    for scan in range(n_scans):
        alpha = scan * (math.pi / n_scans)
        emitters, detectors = _get_positions_numba(radius, alpha, fan_angle, n_detectors, center_x, center_y)

        for i in range(n_detectors):
            x0, y0 = emitters[i]
            x1, y1 = detectors[i]
            value = sinogram[scan, i]
            _backproject_ray_numba(reconstructed, weight_matrix, value, x0, y0, x1, y1, width, height)

        if collect_rmse_per_iteration:
            current = np.zeros_like(reconstructed)
            np.divide(reconstructed, weight_matrix, out=current, where=weight_matrix != 0)

            current = normalize_robust(current)
            rmse_per_iteration.append(calculate_rmse(image_array, current))

        if progress_bar:
            progress_bar.progress(0.5 + (scan + 1) / (2 * n_scans), text="Rekonstrukcja obrazu...")

    np.divide(reconstructed, weight_matrix, out=reconstructed, where=weight_matrix != 0)

    reconstructed = normalize_robust(reconstructed)

    if collect_rmse_per_iteration:
        return display_sinogram, reconstructed, rmse_per_iteration
    return display_sinogram, reconstructed


@njit
def _backproject_ray_numba(reconstructed, weight_matrix, value, x0, y0, x1, y1, width, height):
    points = _bresenham_numba(x0, y0, x1, y1, width, height)
    for i in range(points.shape[0]):
        x = points[i, 0]
        y = points[i, 1]
        reconstructed[y, x] += value
        weight_matrix[y, x] += 1


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