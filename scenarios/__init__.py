from __future__ import annotations

import os
from functools import partial

class FrozenDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        [setattr(self, attr, partial(self.__immutable__, err_msg=attr)) 
         for attr in ("pop", "popitem", "clear", "setdefault", "update")]

    def __immutable__(self, *args, **kwargs) -> TypeError:
        raise TypeError(f"'FrozenDict' object does not support {kwargs.get("err_msg", "this operation")}")
    
    def __copy__(self) -> FrozenDict:
        return FrozenDict(super().copy())
    
    __setitem__ = partial(__immutable__, err_msg="item assignment")
    __delitem__ = partial(__immutable__, err_msg="item deletion")

# import vizdoom as vzd
# scenarios_path = vzd.scenarios_path
scenarios_path = "./scenarios"

# Maps
maps = {
    "map1"  : os.path.join(scenarios_path, "bots_deathmatch_1.cfg"),
    "map1a" : os.path.join(scenarios_path, "bots_deathmatch_1a.cfg"),
    "map1w" : os.path.join(scenarios_path, "bots_deathmatch_1w.cfg"),
    "map2"  : os.path.join(scenarios_path, "bots_deathmatch_2.cfg"),
    "map2s" : os.path.join(scenarios_path, "bots_deathmatch_2s.cfg"),
    "map3"  : os.path.join(scenarios_path, "bots_deathmatch_3.cfg"),
}

# Add in real-time semantic segmentation alias for convenience
for k in list(maps.keys()):
    if "rtss_" not in k:
        maps["rtss_" + k] = maps[k]   
del k

# Definition for model input's representations
input_definitions = {
    0   : 3,    # type 0: 3-channel RGB , coloured game frame
    1   : 1,    # type 1: 1-channel S   , ss (semantic segmentation) mask
    2   : 4,    # type 2: 4-channel RGBS, coloured game frame + ss mask
    3   : 3     # type 3: 3-channel RGB , ss mask mapped to RGB via colour map
}

# Add in string alias for input representation
for i, repr_str in enumerate(["rgb", "ss", "ss_rgb", "srgb"]):
    input_definitions[repr_str] = i

# Make them immutable
maps = FrozenDict(maps)
input_definitions = FrozenDict(input_definitions)

if __name__ == "__main__":
    try:
        maps.pop(0)
    except Exception as e:
        assert isinstance(e, TypeError)
    try:
        del maps["map1"]
    except Exception as e:
        assert isinstance(e, TypeError)
    try:
        maps["map7"] = None
    except Exception as e:
        assert isinstance(e, TypeError)
    print("FrozenDict tests passed", flush=True)