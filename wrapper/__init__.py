import gc
from time import time

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from vizdoom import vizdoom as vzd
from stable_baselines3.common.callbacks import EvalCallback, sync_envs_normalization

from vizdoom_utils import RewardTracker, semseg, semseg_rgb, create_game
from models.replay_memory import ReplayMemory

class DummyArray:
    def __setitem__(self, *args, **kwargs) -> list[object]:
        return None

class DummyMemory:
    def __init__(self) -> None:
        self._ptr = -1
    def add(self, *args, **kwargs) -> None:
        self._ptr += 1

class ReplayMemoryNoFrames:
    def __init__(self, mem: ReplayMemory) -> None:
        mem.frames = DummyArray()
        self._memory = mem
        self._ptr = self._memory._ptr
    def add(self, _, *args, **kwargs) -> None:
        self._memory.add(0, *args, **kwargs)
        self._ptr = self._memory._ptr

class DoomBotDeathMatch(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 35}

    def __init__(self, actions: list[list[bool]], input_shape: tuple[int], game_config, reward_tracker: RewardTracker|None=None, 
                 frame_repeat: int=4, is_eval: bool=False, seed: int=None, input_rep: int=0, set_map: str='', realtime_ss=None,
                 frame_stack: bool=False):
        assert isinstance(frame_repeat, int) and frame_repeat > 0, f"Frame repeat value ({frame_repeat}) must be an integer >= 1"
        
        if not realtime_ss is None:
            print("Warning: realtime_ss not implemented for training", flush=True)
        
        self.frame_stack = False
        if frame_stack:
            if frame_repeat > 1:
                self.step = self.step_frame_stack
                self.frame_stack = True
            else:
                print("Warning: frame stacking with a frame repeat of 1 is meaningless", flush=True)

        super().__init__()
        self.action_space = spaces.Discrete(len(actions), seed=seed)
        self.observation_space = spaces.Box(low=0, high=255, shape=input_shape, dtype=np.uint8)
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.actions = actions
        self.game = create_game(**game_config)
        if not (seed is None):
            self.game.set_seed(seed)
        if set_map:
            self.game.set_doom_map(set_map)
        self.game.new_episode()
        self.game.send_game_command("removebots")
        for _ in range(8):
            self.game.send_game_command('addbot')
        
        self.frame_repeat = frame_repeat
        self.is_eval = is_eval
        if reward_tracker is None:
            self.reward_tracker = RewardTracker(self.game)
        else:
            self.reward_tracker = reward_tracker
        
        self.semseg = semseg

        match input_rep:
            case 1:
                self.extract_frame = lambda state: np.expand_dims(self.semseg(state), axis=0)
            case 2:
                self.extract_frame = lambda state: np.dstack([state.screen_buffer, self.semseg(state)]).transpose(2,0,1)
            case 3:
                self.extract_frame = lambda state: semseg_rgb(state)
            case _:
                self.extract_frame = lambda state: state.screen_buffer.transpose(2,0,1)
    
    def step(self, action_id: int):
        reward = self.game.make_action(self.actions[action_id], self.frame_repeat) + self.reward_tracker.update()
        terminated = self.game.is_episode_finished()
        if self.game.is_player_dead() and not terminated:
            self.game.respawn_player()
            terminated = self.game.is_episode_finished()
            reward -= 1
        truncated = False
        info = {}
        if self.is_eval:
            reward = self.reward_tracker.delta_frag
        state = self.game.get_state()
        observation = self.empty_frame if terminated else self.extract_frame(state)
        return observation, reward, terminated, truncated, info
    
    def step_frame_stack(self, action_id: int):
        stack = [self.empty_frame]
        reward = 0
        for _ in [0] * self.frame_repeat:
            reward += self.game.make_action(self.actions[action_id])
            terminated = self.game.is_episode_finished()
            if self.game.is_player_dead() and not terminated:
                self.game.respawn_player()
                terminated = self.game.is_episode_finished()
                reward -= 1
            if terminated:
                break
            stack.append(self.extract_frame(self.game.get_state()))

        missing_frames = self.frame_repeat - (len(stack) - 1)
        if missing_frames:
            stack += [stack[-1]] * missing_frames
        observation = np.concatenate(stack[1:], axis=0)
        
        if self.is_eval:
            reward = self.reward_tracker.delta_frag
        else:
            reward += self.reward_tracker.update()
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        if not (seed is None):
            self.action_space.seed(seed)
            self.game.set_seed(seed)
        self.game.new_episode()
        self.game.send_game_command("removebots")
        for _ in range(8):
            self.game.send_game_command('addbot')
        self.reward_tracker.reset_last_vars()
        info = {}
        state = self.game.get_state()
        observation = self.empty_frame if state is None else self.extract_frame(state)
        if self.frame_stack:
            observation = np.repeat(observation, self.frame_repeat, axis=0)
        return observation, info

    def render(self):
        pass

    def close(self):
        self.reward_tracker.reset_last_vars()

class DoomBotDeathMatchCapture(DoomBotDeathMatch):
    def __init__(self, actions: list[list[bool]], input_shape: tuple[int], game_config, 
                 reward_tracker: RewardTracker | None = None, frame_repeat: int = 4, is_eval: bool = False, 
                 seed: int = None, input_rep: int = 0, set_map: str='', memsize: int=40_000, smooth_frame: bool=False,
                 realtime_ss: bool=False, only_pos: bool=False, frame_stack: bool=False):
        super().__init__(actions, input_shape, game_config, reward_tracker, frame_repeat, is_eval, seed, input_rep, set_map, realtime_ss, frame_stack)
        if smooth_frame:
            self.smooth_frame = True
            self.step = self.step_no_skip
            memsize *= frame_repeat
        else:
            self.smooth_frame = False
            if self.frame_stack:
                self.step = self.step_no_skip
        if memsize > 0:
            if only_pos:
                self.memory = ReplayMemoryNoFrames(res=(1, 1), ch_num=1, size=memsize, dtypes=[np.uint8, np.uint8, 'f2', 'f2', 'f2'])
            else:
                self.memory = ReplayMemory(res=input_shape[1:], ch_num=4, size=memsize, dtypes=[np.uint8, np.uint8, 'f2', 'f2', 'f2'])
        else:
            self.memory = DummyMemory()
        self.ep_ends = [0]
        self.semseg = semseg
        if realtime_ss and input_rep in [1, 2]:
            import models.ss
            from semantic_segmentation.dataset import ToTensorNormalizeSingleImg
            self.transform = ToTensorNormalizeSingleImg(device=models.ss.device)
            self.cpu = torch.device("cpu")
            models.ss.init_res101()
            def realtime_ss(state):
                rgb_frame = state.screen_buffer
                res = rgb_frame.shape[:2]
                input_tensor = rgb_frame[:120, :, :].transpose(2,0,1)
                input_tensor = torch.from_numpy(input_tensor).to(device=models.ss.device)
                input_tensor = self.transform(input_tensor[None, :, :, :])
                with torch.no_grad():
                    output_tensor = torch.argmax(models.ss.res101(input_tensor)['out'], dim=1)[0, :, :]
                output_tensor *= 20
                output_array = np.zeros(shape=res, dtype=np.uint8)
                output_array[:120, :] = output_tensor.cpu()
                return output_array
            self.semseg = realtime_ss
        self.record_frame = lambda state, ss: np.dstack([state.screen_buffer, ss])
        match input_rep:
            case 1:
                self.extract_frame = lambda state, ss, prev: np.expand_dims(ss, axis=0)
            case 2:
                self.extract_frame = lambda state, ss, prev: prev.transpose(2,0,1)
            case 3:
                self.extract_frame = lambda state, ss, prev: semseg_rgb(state)
            case _:
                self.extract_frame = lambda state, ss, prev: state.screen_buffer.transpose(2,0,1)
        self.fps = []
        self.start_time = time()
    
    def step(self, action_id: int):
        self.game.make_action(self.actions[action_id], self.frame_repeat)
        self.reward_tracker.update()
        terminated = self.game.is_episode_finished()
        if self.game.is_player_dead() and not terminated:
            self.game.respawn_player()
            terminated = self.game.is_episode_finished()
        state = self.game.get_state()
        if terminated:
            observation = self.empty_frame
            self.ep_ends.append(self.memory._ptr + 1)
            self.fps.append((self.ep_ends[-1] - self.ep_ends[-2]) / (time() - self.start_time))
        else:
            ss = self.semseg(state)
            frame_save = self.record_frame(state, ss)
            observation = self.extract_frame(state, ss, frame_save)
        if state is not None:
            self.memory.add(frame_save, state.game_variables[1], action_id, state.game_variables[-3:])
        return observation, self.reward_tracker.delta_frag, terminated, False, {"r" : self.reward_tracker.last_frag}
    
    def step_no_skip(self, action_id: int):
        stack = [self.empty_frame]
        latest_state_to_add = None
        for _ in [0] * self.frame_repeat:
            self.game.make_action(self.actions[action_id], 1)
            terminated = self.game.is_episode_finished()
            if self.game.is_player_dead() and not terminated:
                self.game.respawn_player()
                terminated = self.game.is_episode_finished()
            state = self.game.get_state()
            if terminated:
                observation = self.empty_frame
                self.ep_ends.append(self.memory._ptr + 1)
                self.fps.append((self.ep_ends[-1] - self.ep_ends[-2]) / (time() - self.start_time))
                break
            else:
                ss = self.semseg(state)
                frame_save = self.record_frame(state, ss)
                observation = self.extract_frame(state, ss, frame_save)
                if self.frame_stack:
                    stack.append(observation)
                latest_state_to_add = (frame_save, state.game_variables[1], action_id, state.game_variables[-3:])
                if self.smooth_frame:
                    self.memory.add(*latest_state_to_add)
        self.reward_tracker.update()

        # Only memorize last state to match ViZDoom frame repeat behavior
        if not self.smooth_frame:
            self.memory.add(*latest_state_to_add)
        
        if self.frame_stack:
            missing_frames = self.frame_repeat - (len(stack) - 1)
            if missing_frames:
                stack += [stack[-1]] * missing_frames
            observation = np.concatenate(stack[1:], axis=0)

        return observation, self.reward_tracker.delta_frag, terminated, False, {"r" : self.reward_tracker.last_frag}
    
    def reset(self, seed=None, options=None):
        if not (seed is None):
            self.action_space.seed(seed)
            self.game.set_seed(seed)
        self.start_time = time()
        self.game.new_episode()
        self.game.send_game_command("removebots")
        for _ in range(8):
            self.game.send_game_command('addbot')
        self.reward_tracker.reset_last_vars()
        info = {}
        state = self.game.get_state()
        if state is None:
            observation = self.empty_frame
        else:
            ss = self.semseg(state)
            frame_save = self.record_frame(state, ss)
            observation = self.extract_frame(state, ss, frame_save)
            self.memory.add(frame_save, state.game_variables[1], 0, state.game_variables[-3:])
        if self.frame_stack:
            observation = np.repeat(observation, self.frame_repeat, axis=0)
        return observation, info
    
    def save(self, path_: str):
        if isinstance(self.memory, ReplayMemory):
            np.savez_compressed(path_, 
                                obs=self.memory.frames[:self.memory._ptr+1, :, :, :], 
                                feats=self.memory.features[:self.memory._ptr+1, :],
                                ep_ends=np.array(self.ep_ends[1:], dtype=np.uint64),
                                weapon=self.memory.rewards[:self.memory._ptr+1],
                                fps=self.fps)
            del self.memory, self.ep_ends
            gc.collect()
        elif isinstance(self.memory, DummyMemory):
            np.savez_compressed(path_, 
                                obs=None, 
                                feats=None,
                                ep_ends=np.array(self.ep_ends[1:], dtype=np.uint64),
                                weapon=None,
                                fps=np.array(self.fps, dtype=np.uint64))
        elif isinstance(self.memory, ReplayMemoryNoFrames):
            np.savez_compressed(path_, 
                                obs=None, 
                                feats=self.memory.features[:self.memory._ptr+1, :],
                                ep_ends=np.array(self.ep_ends[1:], dtype=np.uint64),
                                weapon=self.memory.rewards[:self.memory._ptr+1],
                                fps=np.array(self.fps, dtype=np.uint64))
        return self.fps