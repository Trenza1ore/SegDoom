from typing import Dict
from collections import defaultdict

import numpy as np
import vizdoom as vzd

DEFAULT_LABELS_DEF = defaultdict(lambda : 0)
DEFAULT_LABELS_DEF["DoomPlayer"] = 1
USE_ASYNC = False

if USE_ASYNC:
    import asyncio

    async def write_buffer(obj: vzd.Label, buffer: np.ndarray, raw_buffer: np.ndarray, label_def: dict):
        buffer[raw_buffer == obj.value] = label_def[obj.object_name]

    async def semseg(state: vzd.GameState, label_def: Dict[str, int]=DEFAULT_LABELS_DEF) -> np.ndarray:
        raw_buffer: np.ndarray = state.labels_buffer
        buffer:     np.ndarray = np.zeros_like(raw_buffer)
        await asyncio.gather(*[write_buffer(obj, buffer, raw_buffer, label_def) for obj in state.labels])
        return buffer
else:
    def semseg(state: vzd.GameState, label_def: Dict[str, int]=DEFAULT_LABELS_DEF) -> np.ndarray:
        raw_buffer: np.ndarray = state.labels_buffer
        buffer:     np.ndarray = np.zeros_like(raw_buffer)
        for obj in state.labels:
            buffer[raw_buffer == obj.value] = label_def[obj.object_name]
        return buffer