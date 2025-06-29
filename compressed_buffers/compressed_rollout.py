from typing import Literal, Generator, Optional, Union

from collections import namedtuple
from functools import partial
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples, VecNormalize
from compressed_buffers.utils import rle_compress, rle_decompress

CompressionMethods = namedtuple("CompressionMethod", ["compress", "decompress"])

__compression_method_mapping = {
    "rle": CompressionMethods(compress=rle_compress, decompress=rle_decompress)
}


class CompressedRolloutBuffer(RolloutBuffer):
    observations: np.ndarray[object]
    len_arr: np.ndarray
    pos_arr: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        dtypes: Optional[dict] = None,
        normalize_images: bool = True,
        compression_method: Literal["rle"] = "rle"
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.dtypes = dtypes or dict(len_type=np.uint8, pos_type=np.uint16, elem_type=np.uint8)
        self.compress, self.decompress = __compression_method_mapping[compression_method]
        self.compress = partial(__compression_method_mapping[compression_method].compress, **self.dtypes)
        self.normalize_images = normalize_images
        self.reset()

    def reset(self) -> None:
        self.observations = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.len_arr = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.pos_arr  = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Compress everything
        for env in range(self.n_envs):
            l, p, e = self.compress(np.ravel(obs[env]))
            self.len_arr[self.pos, env] = l
            self.pos_arr[self.pos, env] = p
            self.observations[self.pos, env] = e

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "len_arr",
                "pos_arr",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        obs = th.stack([self.reconstruct_obs(i) for i in batch_inds])
        if self.normalize_images:
            obs /= 255.0
        data = (
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(obs, *tuple(map(self.to_torch, data)))
    
    # @lru_cache(maxsize=int(os.environ.get("DECOMP_CACHE_SIZE", 512))) # Doesn't make sense on second thought
    def reconstruct_obs(self, idx: int):
        len_arr = np.frombuffer(self.len_arr[idx], self.dtypes["len_type"])
        pos_arr = np.frombuffer(self.pos_arr[idx], self.dtypes["pos_type"])
        elem_arr = np.frombuffer(self.observations[idx], self.dtypes["elem_type"])
        obs = self.decompress(len_arr, pos_arr, elem_arr, arr_configs=dict(shape=self.obs_shape, dtype=np.float32))
        return th.from_numpy(obs).to(self.device, th.float32)
