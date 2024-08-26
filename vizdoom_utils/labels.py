from typing import Dict
from collections import defaultdict

import numpy as np
import vizdoom as vzd
from matplotlib import cm

CM = cm.jet

USE_ASYNC = False
DEFAULT_LABELS_DEF = defaultdict(lambda : 0)
DEFAULT_LABELS_DEF["Floor/Ceil"] = 0
DEFAULT_LABELS_DEF["Wall"] = 20
DEFAULT_LABELS_DEF["ItemFog"] = 40
DEFAULT_LABELS_DEF["TeleportFog"] = 60
DEFAULT_LABELS_DEF["BulletPuff"] = 80
DEFAULT_LABELS_DEF["Blood"] = 100
DEFAULT_LABELS_DEF["Clip"] = 120
DEFAULT_LABELS_DEF["ShellBox"] = 140
DEFAULT_LABELS_DEF["Shotgun"] = 160
DEFAULT_LABELS_DEF["Medikit"] = 180
DEFAULT_LABELS_DEF["DeadDoomPlayer"] = 200
DEFAULT_LABELS_DEF["DoomPlayer"] = 220
DEFAULT_LABELS_DEF["Self"] = 240

DEFAULT_LABELS_DEF_RGB = defaultdict(lambda : np.asfarray([CM(0.)[:3]]).T.copy())
for obj_name, obj_val in DEFAULT_LABELS_DEF.items():
    DEFAULT_LABELS_DEF_RGB[obj_name] = np.asfarray([CM(obj_val / 240.)[:3]]).T.copy()

def semseg(state: vzd.GameState, label_def: Dict[str, int]=None) -> np.ndarray:
    if label_def is None:
        label_def = DEFAULT_LABELS_DEF
    raw_buffer: np.ndarray = state.labels_buffer
    buffer:     np.ndarray = np.zeros_like(raw_buffer)
    buffer[raw_buffer == 1] = label_def["Wall"]

    if state.labels and "Self" in label_def:
        for obj in state.labels[:-1]:
            buffer[raw_buffer == obj.value] = label_def[obj.object_name]
        
        last_obj = state.labels[-1]
        if last_obj.object_name == "DoomPlayer":
            buffer[raw_buffer == last_obj.value] = label_def["Self"]
        else:
            buffer[raw_buffer == last_obj.value] = label_def[last_obj.object_name]
    else:
        for obj in state.labels:
            buffer[raw_buffer == obj.value] = label_def[obj.object_name]
    
    return buffer

def semseg_rgb(state: vzd.GameState, label_def: Dict[str, int]=None) -> np.ndarray:
    if label_def is None:
        label_def = DEFAULT_LABELS_DEF_RGB
    raw_buffer: np.ndarray = state.labels_buffer
    buffer:     np.ndarray = np.empty((3, *raw_buffer.shape), dtype=float)
    buffer[:, :, :] = label_def[''].reshape((3, 1, 1))
    buffer[:, raw_buffer == 1] = label_def["Wall"]

    if state.labels and "Self" in label_def:
        for obj in state.labels[:-1]:
            buffer[:, raw_buffer == obj.value] = label_def[obj.object_name]
        
        last_obj = state.labels[-1]
        if last_obj.object_name == "DoomPlayer":
            buffer[:, raw_buffer == last_obj.value] = label_def["Self"]
        else:
            buffer[:, raw_buffer == last_obj.value] = label_def[last_obj.object_name]
    else:
        for obj in state.labels:
            buffer[:, raw_buffer == obj.value] = label_def[obj.object_name]
    
    return buffer