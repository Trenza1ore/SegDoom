from time import time

import torch
from gymnasium import Env
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.vec_env import VecEnv

from models import *
from vizdoom_utils import *
from stats import DiscordWebhook
from stats.helper_func import arr_mode

# ============================== What is this ========================================
# Helper functions for running training sessions for an RL agent
# ====================================================================================

class EvalCallbackWithWebhook(EvalCallback):
    def __init__(self, eval_env: Env | VecEnv, callback_on_new_best: BaseCallback | None = None, callback_after_eval: BaseCallback | None = None, n_eval_episodes: int = 5, eval_freq: int = 10000, log_path: str | None = None, best_model_save_path: str | None = None, deterministic: bool = True, render: bool = False, verbose: int = 1, warn: bool = True):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.webhook = None
        self.counter = 1
        self.plot_graph = False
        self.best_mode_reward = 0.0
        self.best_3_modes = [0, 0, 0]
        self.stats = {"Q1" : [], "Q2" : [], "Q3" : [], "Mode" : []}

    def attach_webhook(self, bot: DiscordWebhook, name: str):
        if isinstance(bot, DiscordWebhook):
            self.webhook = bot
            self.name = name
            self.counter = 1
            print("Webhook attached successfully!", flush=True)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            eval_time = time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            eval_time = round(time() - eval_time)

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mode_reward = float(arr_mode(episode_rewards))
            Q1, Q2, Q3 = [float(np.percentile(episode_rewards, i)) for i in (25, 50, 75)]
            IQR = Q3 - Q1
            self.stats["Mode"].append(mode_reward)
            self.stats["Q1"].append(Q1)
            self.stats["Q2"].append(Q2)
            self.stats["Q3"].append(Q3)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            eval_time_hr = eval_time // 3600
            eval_time_hr_str = f"{eval_time_hr}h " if eval_time_hr > 0 else ''
            eval_time %= 3600
            eval_time_mi = eval_time // 60
            eval_time %= 60
            eval_time_str = f"{eval_time_hr_str}{eval_time_mi}m {eval_time}s"
            
            if self.webhook is not None:
                exp_rate_str = f"|exp:{self.model.exploration_rate:.05f}" if "iqn" in self.name else ''
                title_str = f"**{self.name}** [{self.counter:04d}{exp_rate_str}]"

                msg = f"{title_str}: {mode_reward:.2f} (best:{max(mode_reward, self.best_mode_reward):.2f})"
                msg += f"\nEval time: {eval_time_str}"

                if self.counter >= 1 and self.plot_graph:
                    plt.clf()
                    plt.plot(self.evaluations_timesteps, self.stats["Q3"], 'o-', linewidth=1, markersize=3, color="blue", label="Q3")
                    plt.plot(self.evaluations_timesteps, self.stats["Q2"], 'o-', linewidth=1, markersize=3, color="orange", label="Q2")
                    plt.plot(self.evaluations_timesteps, self.stats["Q1"], 'o-', linewidth=1, markersize=3, color="red", label="Q1")
                    plt.plot(self.evaluations_timesteps, self.stats["Mode"], 'k--', linewidth=1, label="mode")
                    # plt.xlim(0, 4_000_000)
                    plt.title(title_str.replace("**", ''))
                    plt.legend()
                    plt.savefig(f"{self.webhook.path}/current.png")
                    if not self.webhook.send_img(msg):
                        self.webhook.send_msg(msg)
                else:
                    self.webhook.send_msg(msg)

            self.counter += 1

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/evaluation_time", eval_time_str)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mode_reward", float(mode_reward))
            self.logger.record("eval/reward_IQR", IQR)
            self.logger.record("eval/reward_Q1", Q1)
            self.logger.record("eval/reward_Q2", Q2)
            self.logger.record("eval/reward_Q3", Q3)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mode_reward >= self.best_3_modes[-1]:
                if mode_reward not in self.best_3_modes:
                    self.best_3_modes.pop()
                    self.best_3_modes.append(mode_reward)
                    self.best_3_modes.sort(reverse=True)
                if self.best_model_save_path is not None:
                    save_path = os.path.join(self.best_model_save_path, f"best_model_{mode_reward}_{self.counter}")
                    self.model.save(save_path)
                self.best_mode_reward = float(max(self.best_3_modes))
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class inf_ep_max:
    """a class created to represent max value as Python 3's int has no such thing
    """
    def __eq__(self, __value: object) -> bool:
        return False
    def __gt__(self, __value: object) -> bool:
        return True
    def __ge__(self, __value: object) -> bool:
        return True
    def __lt__(self, __value: object) -> bool:
        return False
    def __le__(self, __value: object) -> bool:
        return False