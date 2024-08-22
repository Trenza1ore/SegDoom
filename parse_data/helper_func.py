from typing import Callable

import glob

import numpy as np
from tqdm.rich import tqdm
from tqdm.notebook import tqdm as tqdm_n

def load_merged_records(path_: str, load_obs: bool=True, load_pos: bool=True, from_notebook: bool=False,
                        obs_transform: Callable=lambda x: x) -> tuple[list|np.ndarray, list|np.ndarray, list|np.ndarray, list|np.ndarray, list|np.ndarray]:
    obs, pos, ep_ends, weapon, fps, miou = [], [], [], [], [], []
    ep_offset = 0
    matching_files = glob.glob(path_)
    if len(matching_files) < 1:
        raise FileNotFoundError(f"No match for glob expression: {path_}")
    tqdm_config = dict(iterable=matching_files, leave=False, desc=path_)
    tqdm_iter = tqdm_n(**tqdm_config) if from_notebook else tqdm(**tqdm_config)
    for f in tqdm_iter:
        x = np.load(f, allow_pickle=True)
        if load_obs:
            obs.append(x["obs"])
        if load_pos:
            pos.append(obs_transform(x["feats"]))
        if load_obs and "ep_ends" in x:
            ep_ends.append(np.asarray(x["ep_ends"], dtype=np.uint64) + ep_offset)
            ep_offset += len(obs[-1])
        if "weapon" in x:
            weapon.append(x["weapon"])
        if "fps" in x:
            fps.append(x["fps"])
        if "miou" in x:
            miou.append(x["miou"])

    if len(miou) > 0 and all(isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating) for arr in miou):
        miou = np.concatenate(miou, axis=0)
    else:
        miou = []
    return (np.concatenate(obs, axis=0) if load_obs else [], 
            np.concatenate(pos, axis=0) if load_pos else [],
            np.concatenate(ep_ends, axis=0) if ep_ends else [],
            np.concatenate(weapon, axis=0) if weapon else [],
            np.concatenate(fps, axis=0) if fps else [],
            miou
    )