from random import randrange
from time import sleep, time

import vizdoom as vzd
from tqdm import trange
from gymnasium import Env
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.vec_env import VecEnv

from models import *
from stats import *
from vizdoom_utils import *

# ============================== What is this ========================================
# Helper functions for running training sessions for an RL agent
# ====================================================================================

def arr_mode(arr: np.ndarray, keep_shape: bool=False) -> np.ndarray | np.integer:
    bins = np.unique(arr.astype(np.int64) if isinstance(arr, np.ndarray) else np.array(arr, dtype=np.int64))
    counts = np.zeros(bins.shape, dtype=np.uint64)
    for i in range(bins.shape[0]):
        counts[i] = (arr == bins[i]).sum()
    argmax = counts.argmax()
    # result = bins[argmax]                             # Only keeps one mode
    result = (bins[counts == counts[argmax]]).mean()    # Account for multiple modes
    if keep_shape:
        result = np.ones_like(arr) * result
    return result

class EvalCallbackWithWebhook(EvalCallback):
    def __init__(self, eval_env: Env | VecEnv, callback_on_new_best: BaseCallback | None = None, callback_after_eval: BaseCallback | None = None, n_eval_episodes: int = 5, eval_freq: int = 10000, log_path: str | None = None, best_model_save_path: str | None = None, deterministic: bool = True, render: bool = False, verbose: int = 1, warn: bool = True):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.webhook = None
        self.best_mode_reward = 0.0
        self.best_3_modes = [0, 0, 0]
        self.stats = {"Q1" : [], "Q2" : [], "Q3" : [], "Mode" : []}

    def attach_webhook(self, bot: discord_bot, name: str):
        if isinstance(bot, discord_bot):
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
            if self.webhook is not None:
                exp_rate_str = f"|exp:{self.model.exploration_rate:.05f}" if "iqn" in self.name else ''
                title_str = f"**{self.name}** [{self.counter:04d}{exp_rate_str}]"

                eval_time_hr = eval_time // 3600
                eval_time_hr_str = f"{eval_time_hr}h " if eval_time_hr > 0 else ''
                eval_time %= 3600
                eval_time_mi = eval_time // 60
                eval_time %= 60
                eval_time_str = f"{eval_time_hr_str}{eval_time_mi}m {eval_time}s"

                msg = f"{title_str}: {mode_reward:.2f} (best:{max(mode_reward, self.best_mode_reward):.2f})"
                msg += f"\nEval time: {eval_time_str}"

                if self.counter >= 1:
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

def train_agent(game: vzd.vizdoom.DoomGame, 
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                skip_training: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=(lambda x, y: x), res: tuple[int, int]=(128, 72), 
                random_runs: bool=False, ep_max: int=0, save_interval: int=0,
                label_def: dict=None
                ) -> tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]:
    """Runs a training session of an RL agent, with discord webhook support for sending statistics of training scores after each epoch of training.
    Written in an unstructured manner to have marginal (yet existing) performance gains. Deadly corridor is better trained with train_agent_corridor.
    
    Args:
        game (vzd.vizdoom.DoomGame): vizdoom game instance.
        agent (CombatAgent): the RL agent to train.
        action_space (list): action space for the combat model.
        nav_action_space (list): action space for the navigation model.
        skip_training (bool, optional): whether to skip training (for debug only). Defaults to False.
        discord (bool, optional): whether to have discord webhook send training stats and plots after every epoch. Defaults to False.
        epoch_num (int, optional): number of epoches to train (not necessarily reached if ep_max is set). Defaults to 3.
        frame_repeat (int, optional): the number of frames to repeat. Defaults to 4.
        epoch_step (int, optional): minimum number of steps in an epoch (the last episode in an epoch is always finished). Defaults to 500.
        load_epoch (int, optional): legacy option, use the load_models method of agent instance instead. Defaults to -1.
        downsampler (_type_, optional): downsampling algorithm to use, bilinear intepolation is the most balanced but nearest is slightly faster. Defaults to resize_cv_linear.
        res (tuple[int, int], optional): downsampling algorithm's target resolution. Defaults to (128, 72).
        random_runs (bool, optional): whether to include short, purely randomized episodes between epoches, minimum steps is set to 500. Defaults to False.
        ep_max (int, optional): maximum episodes for this training session, checked after every epoch and overwrites epoch_num, 0=disable. Defaults to 0.
        save_interval (int, optional): automatically save the models and replay memory every save_interval epoches, 0=disable. Defaults to 0.

    Returns:
        tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]: returns the agent, game instance and training scores
    """    
    
    if save_interval == 0:
        save_interval = epoch_num+1
    
    if ep_max <= 0:
        ep_max = inf_ep_max()
    
    extract_frame = (lambda state: downsampler(state.screen_buffer, res)) \
        if agent.memory.ch_num < 4 else \
            (lambda state: np.dstack([state.screen_buffer, semseg(state, label_def)]))
        
    all_scores = [[], []]
    train_quartiles = [[], [], [], []]
    if (not skip_training):
        bot = discord_bot(extra=f"deathmatch '{agent.name}' lr={agent.lr:.8f}")
        try:
            start = time()
            if load_epoch < 0:
                epoch_start = 0
            else:
                epoch_start = load_epoch
            
            action_space_size = len(action_space)
            nav_action_space_size = len(nav_action_space)
            
            print("Initial filling of replay memory with random actions")
            terminated = False
            game.new_episode()
            reward_tracker = RewardTracker(game)
            
            for _ in trange(500, miniters=10):
                state = game.get_state()
                game_variables = state.game_variables
                # frame = downsampler(state.screen_buffer, res)
                frame = extract_frame(state)

                # counter-intuitively, this is actually faster than creating a list of names
                is_combat = False
                for label in state.labels[:-1]:
                    if label.object_name == "DoomPlayer":
                        is_combat = True
                        break
                
                if is_combat:
                    action = randrange(0, action_space_size)
                    reward = game.make_action(nav_action_space[action], frame_repeat) + reward_tracker.update()
                else:
                    action = randrange(0, nav_action_space_size)
                    reward = game.make_action(nav_action_space[action], frame_repeat) + reward_tracker.update()
                
                agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                
                terminated = game.is_episode_finished()
                
                if terminated:
                    game.new_episode()
                    reward_tracker.reset_last_vars()
            
            for epoch in range(epoch_start, epoch_num):
                
                if random_runs:
                    print("Filling of replay memory with random actions")
                    terminated = False
                    
                    for _ in trange(500, miniters=10):
                        state = game.get_state()
                        game_variables = state.game_variables
                        # frame = downsampler(state.screen_buffer, res)
                        frame = extract_frame(state)
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        is_combat = False
                        for label in state.labels[:-1]:
                            if label.object_name == "DoomPlayer":
                                is_combat = True
                                break
                        
                        if is_combat:
                            action = randrange(0, action_space_size)
                            reward += game.make_action(action_space[action], frame_repeat)
                        else: 
                            action = randrange(0, nav_action_space_size)
                            reward += game.make_action(nav_action_space[action], frame_repeat)
                        
                        agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                        
                        terminated = game.is_episode_finished()
                        
                        if terminated:
                            game.new_episode()
                            reward_tracker.reset_last_vars()
                    
                    while not terminated:
                        state = game.get_state()
                        frame = extract_frame(state)
                        #frame = np.dstack([state.screen_buffer, semseg(state, label_def)])
                        game_variables = state.game_variables
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        is_combat = False
                        for label in state.labels[:-1]:
                            if label.object_name == "DoomPlayer":
                                is_combat = True
                                break
                        
                        if is_combat:
                            action = randrange(0, action_space_size)
                            reward += game.make_action(action_space[action], frame_repeat)
                        else: 
                            action = randrange(0, nav_action_space_size)
                            reward += game.make_action(nav_action_space[action], frame_repeat)
                        
                        agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                        
                        terminated = game.is_episode_finished()
                
                game.new_episode()
                reward_tracker.reset_last_vars()
                train_scores = []
                train_kills = []
                print(f"\n==========Epoch {epoch+1:3d}==========")
                
                terminated = False
                
                for _ in trange(epoch_step, miniters=10):
                    state = game.get_state()
                    frame = extract_frame(state)
                    # frame = np.dstack([state.screen_buffer, semseg(state, label_def)])
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels[:-1]:
                        if label.object_name == "DoomPlayer":
                            is_combat = True
                            break
                    
                    action = agent.decide_move(frame, is_combat)
                    reward += game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat) + reward_tracker.update()
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                    agent.train()
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        train_scores.append(game.get_total_reward() + reward_tracker.total_reward)
                        train_kills.append(game_variables[1])
                        game.new_episode()
                        reward_tracker.reset_last_vars()
                
                tmp = 0
                while not terminated:
                    state = game.get_state()
                    frame = extract_frame(state)
                    # frame = np.dstack([state.screen_buffer, semseg(state, label_def)])
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels[:-1]:
                        if label.object_name == "DoomPlayer":
                            is_combat = True
                            break
                    
                    action = agent.decide_move(frame, is_combat)
                    reward = game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat) + reward_tracker.update()
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                    agent.train()
                    tmp += 1
                    print(tmp, end="\r")
                    
                    if (terminated := game.is_episode_finished()):
                        train_scores.append(game.get_total_reward() + reward_tracker.total_reward)
                        train_kills.append(game_variables[1])
                        # train_frags.append()
                        reward_tracker.reset_last_vars()
                
                # Save statistics
                all_scores[0].extend(train_scores)
                all_scores[1].extend(train_kills)
                
                stats = plot_stat(train_scores, all_scores, train_quartiles, epoch, agent, bot, epoch_start)
                
                duration = int(time()-start);
                timer = f"{duration//60:d} min {duration%60:d} sec"
                print(timer)
                
                if discord:
                    bot.send_string(stats+'\n'+timer)
                    bot.send_img(epoch)
                
                # Save models after epoch
                agent.save_models(epoch) if not (epoch+1)%save_interval else None
                
                if len(all_scores[0]) >= ep_max:
                    agent.save_models(epoch) if (epoch+1)%save_interval else None
                    bot.send_string("Training Complete!")
                    capture(agent, game, frame_repeat, action_space, nav_action_space, extract_frame, res, 10, reward_tracker=reward_tracker)
                    game.close()
                    return
                
                sleep(1) # pause for 30 seconds to recover from heat
        except Exception as e:
            bot.send_error(e)
            game.close()
            return
    
    bot.send_string("Training Complete!")        
    game.close()
    return (agent, game, all_scores)

