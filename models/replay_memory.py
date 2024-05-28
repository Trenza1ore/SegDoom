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
            "frame"     : (dtypes[0], (size, *res, ch_num)),
            "reward"    : (dtypes[1], (size, ))
        }
        
        if len(dtypes) > 2:
            self.dtype["features"] = (*dtypes[2:], )
            self.feature_num = len(self.dtype["features"])
            if len(set(self.dtype["features"])) == 1:
                self.features = np.zeros((size, self.feature_num), dtype=self.dtype['features'][0])
            else:
                for i in range(self.feature_num):
                    exec(f"self.feature_{i} = np.zeros(size, dtype={self.dtype['features'][i]})")
        else:
            self.feature_num = 0
        
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
        
        self.frames = np.zeros((size, *res, ch_num), dtype=dtypes[0])
        self.rewards = np.zeros(size, dtype=dtypes[1])
        self.actions = np.zeros(size, dtype=np.uint8)
        
        self.__ptr = -1
        self.rng = default_rng()
    
    # functions for ease of checking
    def __len__(self) -> int:
        return self.__ptr + 1
    
    def __str__(self) -> str:
        return f"ReplayMemory(f:{self.dtype['frame']}, r:{self.dtype['reward']})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    # add is replaced by add_filled after the memory has been filled once
    def add(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8,
            features: tuple=None):
        """Add a single state into memory
        """        
        if self.__ptr < self.max_index:
            self.__ptr += 1
        else:
            self.__ptr = 0
            self.add = self.add_filled
            self.replay_p = self.replay_p_filled
            self.replay_p_nav = self.replay_p_nav_filled
        self.frames[self.__ptr, :, :, :] = frame
        self.rewards[self.__ptr] = reward
        self.actions[self.__ptr] = action
        if features:
            self.features[self.__ptr, :] = features
    
    def add_filled(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8,
            features: tuple):
        """Add a single state into memory
        """        
        self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
        self.frames[self.__ptr, :, :, :] = frame
        self.rewards[self.__ptr] = reward
        self.actions[self.__ptr] = action
        if features:
            self.features[self.__ptr, :] = features

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
            scores[self.__ptr:] = 0
            scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        
        # re-roll because selection contains invalid states for training
        # (navigation and combat states mixed together)
        if np.any(self.features[indices, 0] != self.features[indices+1, 0]):
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
            scores[self.__ptr:self.__ptr+self.history_len+1] = 0
            scores[-1] = 0
            scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        
        # re-roll because selection contains invalid states for training
        # (navigation and combat states mixed together)
        if np.any(self.features[indices, 0] != self.features[indices+1, 0]):
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
        scores[self.__ptr:] = 0
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
        scores[self.__ptr:self.__ptr+self.history_len+1] = 0
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
        scores[self.__ptr:] = 0
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
        scores[self.__ptr:self.__ptr+self.history_len+1] = 0
        scores[-1] = 0
        scores /= np.sum(scores)
        indices = self.rng.choice(self.indices, size=n, replace=r, p=scores)
        return indices
    
    # Legacy code kept as reference
    
    # def bulk_add_unsafe(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], 
    #                     actions: np.ndarray[np.uint8], features: np.ndarray, n: int):
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     end = self.__ptr + n
    #     self.frames[self.__ptr:end, :, :, :] = frame
    #     self.rewards[self.__ptr:end] = reward
    #     self.actions[self.__ptr:end] = actions
    #     self.features[self.__ptr:end, :] = features
    
    # def bulk_add(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], 
    #              actions: np.ndarray[np.uint8], features: np.ndarray, n: int):
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     if (free_size := (self.max_size - self.__ptr)) < n:
    #         self.frames[self.__ptr:, :, :, :] = frame[:free_size, :, :, :]
    #         self.rewards[self.__ptr:] = reward[:free_size]
    #         self.actions[self.__ptr:] = actions[:free_size]
    #         self.features[self.__ptr:, :] = features[:free_size, :]
    #         remaining = n - free_size
    #         self.frames[:remaining, :, :, :] = frame[remaining:, :, :, :]
    #         self.rewards[:remaining] = reward[remaining:]
    #         self.actions[:remaining] = actions[remaining:]
    #         self.features[:remaining, :] = features[remaining:, :]
    #         self.__ptr = remaining - 1
    #     else:
    #         end = self.__ptr + n
    #         self.frames[self.__ptr:end, :, :, :] = frame
    #         self.rewards[self.__ptr:end] = reward
    #         self.actions[self.__ptr:end] = actions
    #         self.features[self.__ptr:end, :] = features
    #         self.__ptr = end - 1
    
    # def add_1f(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8,
    #         feature):
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     self.frames[self.__ptr, :, :, :] = frame
    #     self.rewards[self.__ptr] = reward
    #     self.actions[self.__ptr] = action
    #     self.feature_0[self.__ptr] = feature
    
    # def bulk_add_unsafe_1f(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], 
    #                     actions: np.ndarray[np.uint8], features: np.ndarray, n: int):
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     end = self.__ptr + n
    #     self.frames[self.__ptr:end, :, :, :] = frame
    #     self.rewards[self.__ptr:end] = reward
    #     self.actions[self.__ptr:end] = actions
    #     self.feature_0[self.__ptr:end] = features
    
    # def bulk_add_1f(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], 
    #              actions: np.ndarray[np.uint8], features: np.ndarray, n: int):
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     if (free_size := (self.max_size - self.__ptr)) < n:
    #         self.frames[self.__ptr:, :, :, :] = frame[:free_size, :, :, :]
    #         self.rewards[self.__ptr:] = reward[:free_size]
    #         self.actions[self.__ptr:] = actions[:free_size]
    #         self.feature_0[self.__ptr:] = features[:free_size]
    #         remaining = n - free_size
    #         self.frames[:remaining, :, :, :] = frame[remaining:, :, :, :]
    #         self.rewards[:remaining] = reward[remaining:]
    #         self.actions[:remaining] = actions[remaining:]
    #         self.feature_0[:remaining] = features[remaining:]
    #         self.__ptr = remaining - 1
    #     else:
    #         end = self.__ptr + n
    #         self.frames[self.__ptr:end, :, :, :] = frame
    #         self.rewards[self.__ptr:end] = reward
    #         self.actions[self.__ptr:end] = actions
    #         self.feature_0[self.__ptr:end] = features
    #         self.__ptr = end - 1
        
    # def add_2f(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8,
    #         features: tuple):
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     self.frames[self.__ptr, :, :, :] = frame
    #     self.rewards[self.__ptr] = reward
    #     self.actions[self.__ptr] = action
    #     self.feature_0[self.__ptr], self.feature_1[self.__ptr] = features
    
    # def bulk_add_unsafe_2f(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], 
    #                     actions: np.ndarray[np.uint8], features: tuple[np.ndarray], n: int):
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     end = self.__ptr + n
    #     self.frames[self.__ptr:end, :, :, :] = frame
    #     self.rewards[self.__ptr:end] = reward
    #     self.actions[self.__ptr:end] = actions
    #     self.feature_0[self.__ptr:end], self.feature_1[self.__ptr:end] = features
            
    # def bulk_add_2f(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], 
    #              actions: np.ndarray[np.uint8], features: tuple[np.ndarray], n: int):
    #     f0, f1 = features
    #     self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
    #     if (free_size := (self.max_size - self.__ptr)) < n:
    #         self.frames[self.__ptr:, :, :, :] = frame[:free_size, :, :, :]
    #         self.rewards[self.__ptr:] = reward[:free_size]
    #         self.actions[self.__ptr:] = actions[:free_size]
    #         self.feature_0[self.__ptr:] = f0[:free_size]
    #         self.feature_1[self.__ptr:] = f1[:free_size]
    #         remaining = n - free_size
    #         self.frames[:remaining, :, :, :] = frame[remaining:, :, :, :]
    #         self.rewards[:remaining] = reward[remaining:]
    #         self.actions[:remaining] = actions[remaining:]
    #         self.feature_0[:remaining] = f0[remaining:]
    #         self.feature_1[:remaining] = f1[remaining:]
    #         self.__ptr = remaining - 1
    #     else:
    #         end = self.__ptr + n
    #         self.frames[self.__ptr:end, :, :, :] = frame
    #         self.rewards[self.__ptr:end] = reward
    #         self.actions[self.__ptr:end] = actions
    #         self.feature_0[self.__ptr:end] = f0
    #         self.feature_1[self.__ptr:end] = f1
    #         self.__ptr = end - 1
    
    # def replay(self, n: int) -> np.ndarray[np.unsignedinteger]:
    #     return self.rng.integers(self.history_len, self.__ptr, size=n)
    
    # def replay(self, n: int) -> tuple[np.ndarray[np.integer], np.ndarray[np.floating]]:
    #     random_indices = randint(0, self.max_size, size=n)
    #     return (self.frames[random_indices, :, :, :], self.rewards[random_indices])
    
    # def replay_p(self, n: int, r: bool=True) -> tuple[np.ndarray[np.integer], np.ndarray[np.floating]]:
    #     scores = self.rewards - np.min(self.rewards)
    #     scores = scores / np.sum(scores)
    #     random_indices = choice(self.indices, size=n, replace=r, p=scores)
    #     return (self.frames[random_indices, :, :, :], self.rewards[random_indices])