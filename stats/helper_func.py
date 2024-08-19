import numpy as np
from scipy.signal import lfilter

def ewa_smooth(arr: np.ndarray, weight: float) -> np.ndarray:
    return lfilter([1. - weight], [1., -weight], arr, zi=[arr[0]])[0]