import math

import numpy as np

from .geometry import get_positions

try:
    from numba import njit
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator
else:
    NUMBA_AVAILABLE = True


@njit(cache=True)
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


@njit(cache=True)
def _normalize_01_numba(image):
    min_val = image.flat[0]
    max_val = image.flat[0]

    for value in image.flat:
        if value < min_val:
            min_val = value
        elif value > max_val:
            max_val = value

    normalized = np.zeros_like(image)
    if max_val == min_val:
        return normalized

    scale = 1.0 / (max_val - min_val)
    for index, value in np.ndenumerate(image):
        normalized[index] = (value - min_val) * scale
    return normalized


def calculate_rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))


def _normalize_01(image):
    return _normalize_01_numba(np.asarray(image, dtype=np.float64))


def normalize_array(image):
    return _normalize_01(image)


@njit(cache=True)
def _ray_mean_numba(image_array, x0, y0, x1, y1, width, height):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x = x0
    y = y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    total = 0.0
    count = 0

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if 0 <= x < width and 0 <= y < height:
                total += image_array[y, x]
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
                total += image_array[y, x]
                count += 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    if 0 <= x < width and 0 <= y < height:
        total += image_array[y, x]
        count += 1

    if count == 0:
        return 0.0
    return total / count


def _ray_mean(image_array, points):
    if not points:
        return 0.0

    coordinates = np.asarray(points, dtype=np.intp)
    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    return float(image_array[ys, xs].mean())


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
        emitters, detectors = get_positions(radius, alpha, fan_angle, n_detectors, center_x, center_y)

        for i in range(n_detectors):
            x0, y0 = emitters[i]
            x1, y1 = detectors[i]
            value = sinogram[scan, i]
            _backproject_ray_numba(reconstructed, weight_matrix, value, x0, y0, x1, y1, width, height)

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


@njit(cache=True)
def _backproject_ray_numba(reconstructed, weight_matrix, value, x0, y0, x1, y1, width, height):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x = x0
    y = y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if 0 <= x < width and 0 <= y < height:
                reconstructed[y, x] += value
                weight_matrix[y, x] += 1
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if 0 <= x < width and 0 <= y < height:
                reconstructed[y, x] += value
                weight_matrix[y, x] += 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    if 0 <= x < width and 0 <= y < height:
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

