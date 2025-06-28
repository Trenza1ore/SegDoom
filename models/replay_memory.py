import numpy as np
from numpy.random import choice, randint, default_rng, multinomial

class ReplayMemory:
    def __init__(self, res: tuple[int, int]=(240, 320), ch_num: int=3, size: int=40000, nav: bool=True,
                 history_len: int=6, dtypes: list[object]=[np.uint8, np.float32]):
        """A class that implements replay memory for prioritized experience replay.

        Args:
            res (tuple[int, int], optional): resolution in form of (height, width). Defaults to (240, 320).
            ch_num (int): number of colour channels. Defaults to 3.
            size (int): maximum size, starts deleting old memories after maximum reached. Defaults to 40000.
            nav (bool, optional): whether navigation model is used. Defaults to True.
            history_len (int, optional): number of history states. Defaults to 6.
            dtypes (list[object]): data type of frame, reward and (if any) features, datatype for features 
            should be passed in as strings like "np.uint8" or "bool". Defaults to [np.uint8, np.float32].
        """        
        
        self.max_size = size
        self.max_index = size - 1
        self.ch_num = ch_num
        self.history_len = history_len
        self.dtype = {
            "frame"     : (dtypes[0], (size, ch_num, *res)),
            "reward"    : (dtypes[1], (size, ))
        }
        
        if len(dtypes) > 2:
            self.dtype["features"] = (*dtypes[2:], )
            self.feature_num = len(self.dtype["features"])
            self.use_features = True
            if len(set(self.dtype["features"])) == 1:
                self.features = np.zeros((size, self.feature_num), dtype=self.dtype['features'][0])
            else:
                self.features = np.zeros(size, dtype=[(str(i), self.dtype['features'][i]) for i in range(self.feature_num)])
                # for i in range(self.feature_num):
                #     print(f"EXEC: self.feature_{i} = np.zeros(size, dtype={self.dtype['features'][i]})")
                #     self.features.append(eval(f"self.feature_{i}"))
                #     exec(f"self.feature_{i} = np.zeros(size, dtype={self.dtype['features'][i]})")
        else:
            self.feature_num = 0
            self.use_features = False
        
        # No navigation model
        if not nav:
            self.replay_p = self.replay_p_no_check
            self.replay_p_filled = self.replay_p_filled_no_check
        
        # try to minimize memory usage
        if size < 65_536:
            self.indices = np.arange(size, dtype=np.uint16)
        elif size < 4_294_967_296:
            self.indices = np.arange(size, dtype=np.uint32)
        else:
            self.indices = np.arange(size, dtype=np.uint64)
        
        self.frames = np.zeros((size, ch_num, *res), dtype=dtypes[0])
        self.rewards = np.zeros(size, dtype=dtypes[1])
        self.actions = np.zeros(size, dtype=np.uint8)
        
        self._ptr = -1
        self.rng = default_rng()
    
    # functions for ease of checking
    def __len__(self) -> int:
        return self._ptr + 1
    
    def __str__(self) -> str:
        return f"ReplayMemory(f:{self.dtype['frame']}, r:{self.dtype['reward']})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    # add is replaced by add_filled after the memory has been filled once
    def add(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8,
            features: tuple=None):
        """Add a single state into memory
        """        
        if self._ptr < self.max_index:
            self._ptr += 1
        else:
            self._ptr = 0
            self.add = self.add_filled
            self.replay_p = self.replay_p_filled
            self.replay_p_nav = self.replay_p_nav_filled
        self.frames[self._ptr, :, :, :] = frame.transpose(2,0,1)
        self.rewards[self._ptr] = reward
        self.actions[self._ptr] = action
        if self.use_features:
            self.features[self._ptr] = features
    
    def add_filled(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8,
            features: tuple):
        """Add a single state into memory
        """        
        self._ptr = self._ptr + 1 if self._ptr < self.max_index else 0
        self.frames[self._ptr, :, :, :] = frame.transpose(2,0,1)
        self.rewards[self._ptr] = reward
        self.actions[self._ptr] = action
        if self.use_features:
            self.features[self._ptr] = features

    # replay_p is replaced by replay_p_filled after the memory has been filled once
    def replay_p(self, n: int, r: bool=True, scores=0) -> np.ndarray[np.unsignedinteger]:
        """sample states for prioritized experience replay
        """
        if scores is 0: # if scores == None won't work with numpy arrays...
            scores = np.asfarray(self.rewards)
            # Normalize before setting unpickable states to 0
            # Since negative rewards exist
            scores = scores - np.min(scores)
            scores[:self.history_len] = 0
            scores[self._ptr:] = 0
            scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        
        # re-roll because selection contains invalid states for training
        # (navigation and combat states mixed together)
        if np.any(self.features['0'][indices] != self.features['0'][indices+1]):
            return self.replay_p(n, r, scores)
        return indices
    
    def replay_p_filled(self, n: int, r: bool=True, scores=0) -> np.ndarray[np.unsignedinteger]:
        """sample states for prioritized experience replay
        """
        if scores is 0: # if scores == None won't work with numpy arrays...
            scores = np.asfarray(self.rewards)
            # Normalize before setting unpickable states to 0
            # Since negative rewards exist
            scores = scores - np.min(scores)
            scores[:self.history_len] = 0
            scores[self._ptr:self._ptr+self.history_len+1] = 0
            scores[-1] = 0
            scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        
        # re-roll because selection contains invalid states for training
        # (navigation and combat states mixed together)
        if np.any(self.features['0'][indices] != self.features['0'][indices+1]):
            return self.replay_p(n, r, scores)
        return indices
    
    # replay_p_no_check is replaced by replay_p_filled_no_check after the memory has been filled once
    def replay_p_no_check(self, n: int, r: bool=True) -> np.ndarray[np.unsignedinteger]:
        """sample states for prioritized experience replay (doesn't validate selected states)
        """
        scores = np.asfarray(self.rewards)
        # Normalize before setting unpickable states to 0
        # Since negative rewards exist
        scores = scores - np.min(scores)
        scores[:self.history_len] = 0
        scores[self._ptr:] = 0
        scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        return indices
    
    def replay_p_filled_no_check(self, n: int, r: bool=True) -> np.ndarray[np.unsignedinteger]:
        """sample states for prioritized experience replay (doesn't validate selected states)
        """
        scores = np.asfarray(self.rewards)
        # Normalize before setting unpickable states to 0
        # Since negative rewards exist
        scores = scores - np.min(scores)
        scores[:self.history_len] = 0
        scores[self._ptr:self._ptr+self.history_len+1] = 0
        scores[-1] = 0
        scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        return indices
    
    # replay_p_nav is replaced by replay_p_nav_filled after the memory has been filled once
    def replay_p_nav(self, n: int, r: bool=True) -> np.ndarray[np.unsignedinteger]:
        """sample states for prioritized experience replay (for navigation model)
        """
        scores = np.asfarray(self.rewards)
        # Normalize before setting unpickable states to 0
        # Since negative rewards exist
        scores = scores - np.min(scores)
        scores[self.features[:, 0]] = 0
        scores[:self.history_len] = 0
        scores[self._ptr:] = 0
        scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        return indices
    
    def replay_p_nav_filled(self, n: int, r: bool=True) -> np.ndarray[np.unsignedinteger]:
        """sample states for prioritized experience replay (for navigation model)
        """
        scores = np.asfarray(self.rewards)
        # Normalize before setting unpickable states to 0
        # Since negative rewards exist
        scores = scores - np.min(scores)
        scores[self.features[:, 0]] = 0
        scores[:self.history_len] = 0
        scores[self._ptr:self._ptr+self.history_len+1] = 0
        scores[-1] = 0
        scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        return indices
