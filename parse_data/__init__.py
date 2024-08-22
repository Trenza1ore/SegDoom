from typing import Callable

import glob

import numpy as np
from tqdm.rich import tqdm

def load_merged_records(path_: str, load_obs: bool=True, load_pos: bool=True, obs_transform: Callable=lambda x: x):
    obs, pos, ep_ends, weapon, fps, miou = [], [], [], [], [], []
    ep_offset = 0
    matching_files = glob.glob(path_)
    if len(matching_files) < 1:
        raise FileNotFoundError(f"No match for glob expression: {path_}")
    for f in tqdm(matching_files):
        x = np.load(f)
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

    if len(miou) == 0 or any(np.issubdtype(arr.dtype, np.integer) for arr in miou):
        miou = []
    else:
        miou = np.concatenate(miou, axis=0)
    return (np.concatenate(obs, axis=0) if load_obs else [], 
            np.concatenate(pos, axis=0) if load_pos else [],
            np.concatenate(ep_ends, axis=0) if ep_ends else [],
            np.concatenate(weapon, axis=0) if weapon else [],
            np.concatenate(fps, axis=0) if fps else [],
            miou
    )