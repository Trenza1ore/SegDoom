from typing import Optional

import warnings
import torch
import numpy as np

warnings.filterwarnings(action="ignore", message="The given NumPy array is not writable.*", category=UserWarning)

_unsigned_int_types = [np.uint8, np.uint16, np.uint32, np.uint64, np.uint128, np.uint256]
_signed_int_types =[np.int8, np.int16, np.int32, np.int64, np.int128, np.int256]
_max_val_lookup = {dtype: np.iinfo(dtype).max for dtype in (_unsigned_int_types + _signed_int_types)}

def rle_compress(arr: np.ndarray, len_type: np.dtype = np.uint16, pos_type: np.dtype = np.uint16,
                 elem_type: np.dtype = np.uint8) -> tuple[bytes, bytes, bytes]:
    """RLE Compression, credits:
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi/32681075#32681075
    """
    n = len(arr)
    if n == 0:
        return (None, None, None)
    else:
        y = arr[1:] != arr[:-1]
    idx_arr = np.append(np.where(y), n - 1)
    len_arr = np.diff(np.append(-1, idx_arr))
    pos_arr = np.cumsum(np.append(0, len_arr))[:-1]
    return (len_arr.astype(len_type, copy=False).tobytes(),
            pos_arr.astype(pos_type, copy=False).tobytes(),
            arr[idx_arr].astype(elem_type, copy=False).tobytes())

def rle_decompress(len_arr: np.ndarray, pos_arr: np.ndarray, elements: np.ndarray, arr_configs: dict) -> np.ndarray:
    """RLE Decompression"""
    sum_lengths = len_arr.sum()
    run_indices = np.repeat(np.arange(len(len_arr)), len_arr)

    # Compute indices using vectorized operations
    cumulative_starts = np.concatenate([
        np.array([0], dtype=np.uint8),
        np.cumsum(len_arr, axis=0)[:-1]
    ])
    offsets = np.arange(sum_lengths) - cumulative_starts[run_indices]
    indices = np.repeat(pos_arr, len_arr) + offsets

    # Create values and assign to output tensor
    values = np.repeat(elements, len_arr)
    arr_reconstructed = np.empty(**arr_configs)
    arr_reconstructed[indices] = values
    return arr_reconstructed

def rle_decompress_t(length_tensor: torch.Tensor, pos_tensor: torch.Tensor, elements_tensor: torch.Tensor,
                     device: torch.device, arr_configs: dict) -> np.ndarray:
    """RLE Decompression"""
    sum_lengths = length_tensor.sum().item()
    run_indices = torch.repeat_interleave(torch.arange(len(length_tensor), dtype=torch.int32, device=device),
                                          length_tensor)

    # Compute indices using vectorized operations
    cumulative_starts = torch.cat([
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.cumsum(length_tensor, dim=0, dtype=torch.int32)[:-1]
    ])
    offsets = torch.arange(sum_lengths, dtype=torch.int32, device=device) - cumulative_starts[run_indices]
    indices = torch.repeat_interleave(pos_tensor, length_tensor) + offsets

    # Create values and assign to output tensor
    values = torch.repeat_interleave(elements_tensor, length_tensor)
    tensor_shape = arr_configs.pop("shape")
    arr_reconstructed = torch.empty(tensor_shape, device=device, **arr_configs)
    arr_reconstructed[indices] = values
    return arr_reconstructed

def _downcast_unsigned(arr: np.ndarray) -> np.ndarray:
    max_val = arr.max()
    for dtype in _unsigned_int_types:
        if max_val <= _max_val_lookup[dtype]:
            return arr.astype(dtype)
    raise NotImplementedError(f"Unknown dtype: {arr.dtype}")

def _determine_optimal_shape(arr_len: int, dtype: np.dtype = np.uint8) -> tuple[int, int, int]:
    max_col = _max_val_lookup[dtype] - 1
    max_row = arr_len // max_col
    remainder = 0

    # Try to pack in equal-length slices
    if not arr_len % max_col:
        return max_row, max_col, remainder
    if not arr_len % max_row:
        max_col = arr_len // max_row
        return max_row, max_col, remainder

    # Fine, guess last row is a bit shorter...
    remainder = arr_len - (max_row * max_col)
    return max_row, max_col, remainder

if __name__ == "__main__":
    import time
    import random
    import tqdm
    from tqdm.rich import trange
    warnings.filterwarnings(action="ignore", category=tqdm.TqdmExperimentalWarning)

    np.random.seed(round(time.time()))
    
    ct, nt, tt = [], [], []
    for i in trange(100):
        # Prepare test data
        lengths = np.random.randint(5, 256, 7373)
        arr_ref = np.concatenate([[random.randrange(0, 256)] * i for i in lengths], dtype=np.uint8, casting="unsafe")[:36864]
        arr = arr_ref.reshape((144, 256)).copy().ravel()
        assert np.all(arr.reshape((144, 256)) == arr_ref.reshape((144, 256)))
        
        # Datatypes
        len_type = np.uint16
        pos_type = np.uint16
        elem_type = np.uint8
        dst_type = torch.float32
        device = torch.device("cuda:0")

        # Compress
        t = time.time()
        len_byte, pos_byte, elem_byte = rle_compress(arr)
        t = round((time.time() - t) * 1000000)
        ct.append(t)
        # print(f"RLE-C latency: {t} μs")

        # Decompress
        t = time.time()
        if i % 2:
            length_t = torch.from_numpy(np.frombuffer(len_byte, len_type)).to(device, dtype=torch.int32)
            pos_t = torch.from_numpy(np.frombuffer(pos_byte, pos_type)).to(device, dtype=torch.int32)
            element_t = torch.from_numpy(np.frombuffer(elem_byte, elem_type)).to(device, dtype=dst_type)
            arr_reconstructed_t = rle_decompress_t(length_t, pos_t, element_t, device=device,
                                                arr_configs=dict(dtype=dst_type, shape=arr.shape))
            x = arr_reconstructed_t / 255
            t = round((time.time() - t) * 1000000)
            tt.append(t)
            assert np.all(arr == arr_reconstructed_t.cpu().numpy())
            # print(f"RLE-DT latency: {t} μs")
        else:
            length = np.frombuffer(len_byte, len_type)
            pos = np.frombuffer(pos_byte, pos_type)
            element = np.frombuffer(elem_byte, elem_type)
            arr_reconstructed = rle_decompress(length, pos, element, arr_configs=dict(dtype=elem_type, shape=arr.shape))
            x = torch.from_numpy(arr_reconstructed).to(device, dtype=dst_type) / 255
            t = round((time.time() - t) * 1000000)
            nt.append(t)
            assert np.all(arr == arr_reconstructed)
            # print(f"RLE-DN latency: {t} μs")

    print(f"NumPy: {np.mean(nt)} +/- {np.std(nt)}\nTorch: {np.mean(tt)} +/- {np.std(tt)}")
    # import matplotlib.pyplot as plt
    # size = (144, 256)
    # plt.subplot(211)
    # plt.imshow(arr.reshape(size))
    # plt.subplot(212)
    # plt.imshow(arr_reconstructed.reshape(size))
    # plt.show()
