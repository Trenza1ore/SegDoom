import numpy as np

def rle_compress(arr: np.ndarray) -> np.ndarray:
    """RLE Compression, credits:
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi/32681075#32681075
    """
    n = len(arr)
    if n == 0:
        return (None, None, None)
    else:
        y = arr[1:] != arr[:-1]
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return(z, p, arr[i])

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

if __name__ == "__main__":
    import time
    import random
    import matplotlib.pyplot as plt

    # Prepare test data
    np.random.seed(round(time.time()))
    lengths = np.random.randint(5, 256, 7373)
    arr = np.concatenate([[random.randrange(0, 256)] * i for i in lengths], dtype=np.uint8, casting="unsafe")[:36864]
    
    # Compress
    t = time.time()
    length_tensor, pos_tensor, elements_tensor = rle_compress(arr)
    print(f"RLE-C latency: {round((time.time() - t) * 1000000)} μs")
    t = time.time()
    arr_reconstructed = rle_decompress(length_tensor, pos_tensor, elements_tensor, arr_configs=dict(
        dtype=arr.dtype, shape=arr.shape)
    )
    print(f"RLE-D latency: {round((time.time() - t) * 1000000)} μs")
    print(f"Correctness: {np.all(arr == arr_reconstructed)}")
    size = (144, 256)
    plt.subplot(211)
    plt.imshow(arr.reshape(size))
    plt.subplot(212)
    plt.imshow(arr_reconstructed.reshape(size))
    plt.show()
