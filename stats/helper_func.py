import numpy as np
from scipy.signal import lfilter

def ewa_smooth(arr: np.ndarray, weight: float) -> np.ndarray:
    return lfilter([1. - weight], [1., -weight], arr, zi=[arr[0]])[0]

def arr_mode(arr: np.ndarray, keep_shape: bool=False, no_tie: bool=False) -> np.ndarray | np.integer:
    bins = np.unique(arr.astype(np.int64) if isinstance(arr, np.ndarray) else np.array(arr, dtype=np.int64))
    counts = np.zeros(bins.shape, dtype=np.uint64)
    for i in range(bins.shape[0]):
        counts[i] = (arr == bins[i]).sum()
    argmax = counts.argmax()
    if no_tie:
        result = bins[argmax]                               # Only considers one mode
    else:
        result = (bins[counts == counts[argmax]]).mean()    # Accounts for tying modes
    if keep_shape:
        result = np.ones_like(arr) * result
    return result