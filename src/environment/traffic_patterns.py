import numpy as np
import time


import numpy as np


def _gaussian(hour: float, center: float, width: float) -> float:
    return np.exp(-((hour - center) ** 2) / (2 * width**2))


def get_arrival_rate(hour: float) -> np.ndarray:
    morning_rush = _gaussian(hour, 8.0, 1.5)
    evening_rush = _gaussian(hour, 17.0, 1.5)
    baseline = 0.1

    rate_NS = baseline + 0.8 * morning_rush + 0.4 * evening_rush
    rate_EW = baseline + 0.4 * morning_rush + 0.8 * evening_rush

    return np.array([rate_NS, rate_NS, rate_EW, rate_EW])
