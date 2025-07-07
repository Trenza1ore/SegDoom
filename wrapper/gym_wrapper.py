import gc
import warnings
from time import time
from collections import deque

import torch
import numpy as np
import gymnasium as gym

from vizdoom_utils.labels import DEFAULT_LABELS_DEF
from vizdoom_utils import RewardTracker, semseg, semseg_rgb, create_game
from sb3_extra_buffers.recording import DummyRecordBuffer, FramelessRecordBuffer, RecordBuffer
from sb3_extra_buffers.recording.utils import force_alloc_mem


class DoomBotDeathMatch(gym.Env):

    metadata = {"render_modes": ["offscreen", "human", "rgb_array"], "render_fps": 35}

    def __init__(self, render_mode: str=None, actions: list[list[bool]]=[[True], [False]],
                 input_shape: tuple[int]=(3, 144, 256), game_config={}, reward_tracker: RewardTracker|dict|None=None,
                 frame_repeat: int=4, is_eval: bool=False, seed: int=None, input_rep: int=0, set_map: str='',
                 realtime_ss=None, frame_stack: bool=False, buffer_size: int=-1, n_updates: int=None, bot_num: int=8):
        assert isinstance(frame_repeat, int) and frame_repeat > 0, \
            f"Frame repeat value ({frame_repeat}) must be an integer >= 1"
        assert (isinstance(buffer_size, int) and buffer_size >= 1) if frame_stack else True, \
            f"Buffer size ({buffer_size}) must be an integer >= 1"

        super().__init__()

        warnings.filterwarnings(action="ignore", category=UserWarning)

        if realtime_ss:
            warnings.warn("realtime_ss not implemented for DoomBotDeathMatch as of now and is ignored", FutureWarning)

        if n_updates is None:
            n_updates = buffer_size

        self.frame_stack = False
        self.frame_buffer = None
        if frame_stack:
            if buffer_size > 1:
                self.frame_stack = True
                self.frame_buffer = deque(maxlen=buffer_size)
                self.step = self.step_frame_stack
            else:
                warnings.warn("frame stacking with a buffer size of 1 is meaningless", UserWarning)

        self.render_mode = "offscreen"
        self.action_space = gym.spaces.Discrete(len(actions), seed=seed)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=input_shape, dtype=np.uint8)
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.actions = actions
        self.bot_num = bot_num
        self.frame_repeat = frame_repeat
        self.buffer_size = buffer_size
        self.is_eval = is_eval
        self.n_updates = n_updates
        self.update_iter = tuple([])
        if n_updates > 0:
            self.update_iter = tuple(range(n_updates))
        self.semseg = semseg

        if render_mode in {"human", "rgb_array"}:
            self.render_mode = render_mode
        game_config["visibility"] = render_mode == "human"
        self.game = create_game(**game_config)

        if seed is not None:
            self.game.set_seed(seed)

        if set_map:
            self.game.set_doom_map(set_map)

        if reward_tracker is None:
            self.reward_tracker = RewardTracker(self.game)
        elif isinstance(reward_tracker, dict):
            self.reward_tracker = RewardTracker(self.game, **reward_tracker)
        else:
            self.reward_tracker = reward_tracker

        match input_rep:
            case 1:
                self.extract_frame = lambda state: np.expand_dims(self.semseg(state), axis=0)
            case 2:
                self.extract_frame = lambda state: np.dstack([state.screen_buffer, self.semseg(state)]).transpose(
                    2, 0, 1)
            case 3:
                self.extract_frame = lambda state: semseg_rgb(state)
            case _:
                self.extract_frame = lambda state: state.screen_buffer.transpose(2, 0, 1)

    def step(self, action_id: int):
        reward = self.game.make_action(self.actions[action_id], self.frame_repeat) + self.reward_tracker.update()
        terminated = self.game.is_episode_finished()
        if self.game.is_player_dead() and not terminated:
            self.game.respawn_player()
            terminated = self.game.is_episode_finished()
            reward -= self.reward_tracker.death_penalty()
        truncated = terminated
        info = {}
        if self.is_eval:
            reward = self.reward_tracker.delta_frag
        state = self.game.get_state()
        observation = self.empty_frame if terminated else self.extract_frame(state)
        return observation, reward, terminated, truncated, info

    def step_frame_stack(self, action_id: int):
        reward = 0
        for t in self.update_iter:
            reward += self.game.make_action(self.actions[action_id], self.frame_repeat)
            terminated = self.game.is_episode_finished()
            if self.game.is_player_dead() and not terminated:
                self.game.respawn_player()
                terminated = self.game.is_episode_finished()
                reward -= self.reward_tracker.death_penalty()
            # Early termination
            if terminated:
                missing_frames = self.n_updates - t
                for state in [self.frame_buffer[-1]] * missing_frames:
                    self.frame_buffer.append(state)
                break
            self.frame_buffer.append(self.extract_frame(self.game.get_state()))

        observation = np.concatenate(self.frame_buffer, axis=0)

        reward += self.reward_tracker.update()
        if self.is_eval:
            reward = self.reward_tracker.delta_frag
        truncated = terminated
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.game.set_seed(seed)

        self.game.new_episode()
        self.game.send_game_command("removebots")
        for _ in range(self.bot_num):
            self.game.send_game_command('addbot')
        self.reward_tracker.reset_last_vars()
        info = {}
        state = self.game.get_state()
        observation = self.empty_frame if state is None else self.extract_frame(state)
        if self.frame_stack:
            self.frame_buffer.clear()
            self.frame_buffer.extend([observation] * self.buffer_size)
            observation = np.repeat(observation, self.buffer_size, axis=0)
        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            state = self.game.get_state()
            return self.empty_frame if state is None else self.extract_frame(state)

    def close(self):
        self.reward_tracker.reset_last_vars()
        self.game.close()


class DoomBotDeathMatchCapture(DoomBotDeathMatch):
    def __init__(self, render_mode: str=None, force_alloc: object=None, label_def=DEFAULT_LABELS_DEF,
                 actions: list[list[bool]]=[[True], [False]], input_shape: tuple[int]=(3, 144, 256), game_config={}, 
                 reward_tracker: RewardTracker | dict | None = None, frame_repeat: int = 4, is_eval: bool = False, 
                 seed: int = None, input_rep: int = 0, set_map: str='', memsize: int=40_000, smooth_frame: bool=False,
                 realtime_ss: bool=False, frame_stack: bool=False, buffer_size: int=-1, n_updates: int=None, bot_num: int=8, 
                 only_pos: bool=False, measure_miou: bool=True, dtypes: tuple[np.dtype]=(np.float64, np.int64, np.uint64)):

        super().__init__(render_mode, actions, input_shape, game_config, reward_tracker, frame_repeat, is_eval, 
                         seed, input_rep, set_map, None, frame_stack, buffer_size, n_updates, bot_num)
        if smooth_frame:
            self.frame_repeat_iter = tuple([False] * (self.frame_repeat - 1) + [self.frame_buffer is not None])
            self.step = self.step_smooth
            memsize *= self.frame_repeat * self.n_updates
        else:
            self.frame_repeat_iter = None
            if self.frame_stack:
                self.step = self.step_stack
        if memsize > 0:
            if only_pos:
                self.memory = FramelessRecordBuffer(size=memsize, dtypes=[np.uint8, np.uint8, 'f2', 'f2', 'f2'])
            else:
                self.memory = RecordBuffer(res=input_shape[1:], ch_num=4, size=memsize, 
                                           dtypes=[np.uint8, np.uint8, 'f2', 'f2', 'f2'])
            if force_alloc is not None:
                force_alloc_mem(self.memory, force_alloc)
        else:
            self.memory = DummyRecordBuffer()

        self.ep_ends = [0]
        self.semseg = semseg
        self.scores = []
        self.frame_iou = []
        self.episode_miou = []
        self.episode_iou = []
        self.measure_miou = measure_miou

        if realtime_ss and input_rep in [1, 2]:
            import models.ss
            from semantic_segmentation.dataset import ToTensorNormalizeSingleImg
            self.transform = ToTensorNormalizeSingleImg(device=models.ss.device)
            self.cpu = torch.device("cpu")
            models.ss.init_res101()

            ss_classes = sorted(label_def.values())

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
                if self.measure_miou:
                    self.frame_iou.append(models.ss.calculate_iou(output_array, semseg(state), num_classes=ss_classes))
                return output_array
            self.semseg = realtime_ss
        else:
            if measure_miou:
                warnings.warn(f"Warning: measure_miou={measure_miou} has no effect since no real-time " + \
                              f"semantic segmentation would take place (input_rep={input_rep})", UserWarning)
                self.measure_miou = False

        self.record_frame = lambda state, ss: np.dstack([state.screen_buffer, ss])
        match input_rep:
            case 1:
                self.extract_frame = lambda state, ss, prev: np.expand_dims(ss, axis=0)
            case 2:
                self.extract_frame = lambda state, ss, prev: prev.transpose(2, 0, 1)
            case 3:
                self.extract_frame = lambda state, ss, prev: semseg_rgb(state)
            case _:
                self.extract_frame = lambda state, ss, prev: state.screen_buffer.transpose(2, 0, 1)
        self.fps = []
        self.start_time = time()
        if len(dtypes) == 3:
            self.ftype, self.itype, self.utype = dtypes
        else:
            self.ftype = self.itype = self.utype = dtypes[0]

    def episode_terminates(self):
        self.ep_ends.append(self.memory._ptr + 1)
        self.fps.append((self.ep_ends[-1] - self.ep_ends[-2]) / (time() - self.start_time))
        self.scores.append(self.reward_tracker.last_frag)
        if self.measure_miou:
            self.episode_iou.extend(self.frame_iou)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.episode_miou.append(np.nanmean(np.nanmean(self.frame_iou, axis=1)))
            self.frame_iou.clear()

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
            self.episode_terminates()
        else:
            ss = self.semseg(state)
            frame_save = self.record_frame(state, ss)
            observation = self.extract_frame(state, ss, frame_save)
        if state is not None:
            self.memory.add(frame_save, state.game_variables[1], action_id, state.game_variables[-3:])
        return observation, self.reward_tracker.delta_frag, terminated, terminated, {"r": self.reward_tracker.last_frag}

    def step_stack(self, action_id: int):
        latest_state_to_add = False
        for t in self.update_iter:
            self.game.make_action(self.actions[action_id], self.frame_repeat)
            terminated = self.game.is_episode_finished()
            if self.game.is_player_dead() and not terminated:
                self.game.respawn_player()
                terminated = self.game.is_episode_finished()
            state = self.game.get_state()
            # Early termination
            if terminated:
                missing_frames = self.n_updates - t
                self.frame_buffer.extend([self.frame_buffer[-1]] * missing_frames)
                self.episode_terminates()
                break
            else:
                ss = self.semseg(state)
                frame_save = self.record_frame(state, ss)
                observation = self.extract_frame(state, ss, frame_save)
                self.frame_buffer.append(observation)
                latest_state_to_add = (frame_save, state.game_variables[1], action_id, state.game_variables[-3:])
        self.reward_tracker.update()

        # Only memorize last state to match ViZDoom frame repeat behavior
        if latest_state_to_add:
            self.memory.add(*latest_state_to_add)

        observation = np.concatenate(self.frame_buffer, axis=0)

        return observation, self.reward_tracker.delta_frag, terminated, terminated, {"r": self.reward_tracker.last_frag}

    def step_smooth(self, action_id: int):
        terminated = False
        observation = self.empty_frame
        for t in self.update_iter:
            for tt in self.frame_repeat_iter:
                self.game.make_action(self.actions[action_id])
                terminated = self.game.is_episode_finished()
                if self.game.is_player_dead() and not terminated:
                    self.game.respawn_player()
                    terminated = self.game.is_episode_finished()
                state = self.game.get_state()
                # Early termination
                if terminated:
                    if self.frame_buffer is not None:
                        missing_frames = self.n_updates - t
                        self.frame_buffer.extend([self.frame_buffer[-1]] * missing_frames)
                    self.episode_terminates()
                    break
                else:
                    ss = self.semseg(state)
                    frame_save = self.record_frame(state, ss)
                    observation = self.extract_frame(state, ss, frame_save)
                    if tt:
                        self.frame_buffer.append(observation)
                    self.memory.add(frame_save, state.game_variables[1], action_id, state.game_variables[-3:])
            if terminated:
                break
        self.reward_tracker.update()

        if self.frame_buffer is not None:
            observation = np.concatenate(self.frame_buffer, axis=0)

        return observation, self.reward_tracker.delta_frag, terminated, terminated, {"r": self.reward_tracker.last_frag}

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if not (seed is None):
            self.action_space.seed(seed)
            self.game.set_seed(seed)

        self.start_time = time()
        self.game.new_episode()
        self.game.send_game_command("removebots")
        for _ in range(self.bot_num):
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
            self.frame_buffer.clear()
            self.frame_buffer.extend([observation] * self.buffer_size)
            observation = np.repeat(observation, self.buffer_size, axis=0)

        if self.measure_miou:
            self.frame_iou.clear()
        return observation, info

    def save(self, path_: str):
        if isinstance(self.memory, RecordBuffer):
            np.savez_compressed(path_, 
                                obs=self.memory.frames[:self.memory._ptr+1, :, :, :], 
                                feats=self.memory.features[:self.memory._ptr+1, :],
                                ep_ends=np.array(self.ep_ends[1:], dtype=self.utype),
                                weapon=self.memory.rewards[:self.memory._ptr+1],
                                actions=self.memory.actions[:self.memory._ptr+1],
                                fps=np.array(self.fps, dtype=self.utype),
                                scores=np.array(self.scores, dtype=self.itype),
                                miou=np.array(self.episode_miou, dtype=self.ftype) if self.measure_miou else None,
                                iou=np.array(self.episode_iou, dtype=self.ftype) if self.measure_miou else None)
            del self.memory, self.ep_ends
            gc.collect()
        elif isinstance(self.memory, DummyRecordBuffer):
            np.savez_compressed(path_, 
                                obs=None, 
                                feats=None,
                                ep_ends=np.array(self.ep_ends[1:], dtype=self.utype),
                                weapon=None,
                                actions=None,
                                fps=np.array(self.fps, dtype=self.utype),
                                scores=np.array(self.scores, dtype=self.itype),
                                miou=np.array(self.episode_miou, dtype=self.ftype) if self.measure_miou else None,
                                iou=np.array(self.episode_iou, dtype=self.ftype) if self.measure_miou else None)
        elif isinstance(self.memory, FramelessRecordBuffer):
            np.savez_compressed(path_, 
                                obs=None, 
                                feats=self.memory.features[:self.memory._ptr+1, :],
                                ep_ends=np.array(self.ep_ends[1:], dtype=self.utype),
                                weapon=self.memory.rewards[:self.memory._ptr+1],
                                actions=self.memory.actions[:self.memory._ptr+1],
                                fps=np.array(self.fps, dtype=self.utype),
                                scores=np.array(self.scores, dtype=self.itype),
                                miou=np.array(self.episode_miou, dtype=self.ftype) if self.measure_miou else None,
                                iou=np.array(self.episode_iou, dtype=self.ftype) if self.measure_miou else None)
        return self.fps
