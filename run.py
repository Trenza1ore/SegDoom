import os
import gc
import itertools as it
from time import sleep, time

from tqdm.rich import tqdm
import vizdoom as vzd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, Adagrad
from stable_baselines3 import PPO, HerReplayBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO, IQN

from models import *
from vizdoom_utils import *
from wrapper import DoomBotDeathMatch
from models.training_procedure import *

# ============================== What is this ========================================
# The main program to run training sessions
# ====================================================================================

# Deathmatch Scenarios
deathmatch_bot = os.path.join(vzd.scenarios_path, "deathmatch_simple.cfg")

# Tips for config options
# save_validation should be switch to True at least in the first run to check the size of each save
# format for a task: 
# (scenario_config, learning rate, target number for episodes to train, save interval, unique id)

bot_args = " ".join([
    "-host 1",
    "-deathmatch",
    "+viz_nocheat 0",
    "+cl_run 1",
    "+name AGENT",
    "+colorset 0",
    "+sv_forcerespawn 1",
    "+sv_respawnprotect 1",
    # "+sv_spawnfarthest 1",
    # "+sv_losefrag 1",
    "+sv_nocrouch 1",
    "+sv_noexit 1"
])

bot_args_eval = " ".join([
    "-host 1",
    "-deathmatch",
    "+viz_nocheat 0",
    "+cl_run 1",
    "+name AGENT",
    "+colorset 0",
    "+sv_forcerespawn 1",
    "+sv_respawnprotect 1",
    # "+sv_losefrag 1",
    "+sv_nocrouch 1",
    "+sv_noexit 1"
])

# config
task_idx = 0
save_validation = False
use_recurrency = True
global_framestack = True
warmup_episodes = 200
warmup_n_envs = 20
frame_repeat = 4

input_rep_ch_num = {
    0   : 3,    # RGB
    1   : 1,    # Semantic Segmentation Mask
    2   : 4,    # RGBS (S = Semantic Segmentation Mask)
    3   : 3,    # Semantic Segmentation as RGB
}

# config, lr, ep_max, save_interval, name, input_rep, n_envs, seed, framestack, model
# framestack: -1 = global behavior, 0 = Off, 1 = On
tasks = [
    # 1-3 (PPO for lr=3e-4)
    (deathmatch_bot, 3e-4,      1, 1, "ss_rgb_3e-4",    2, 4, 2050808, -1, "PPO", None),    # 1
    (deathmatch_bot, 3e-4,      1, 1, "rgb_3e-4",       0, 4, 2050808, -1, "PPO", None),    # 2
    (deathmatch_bot, 3e-4,      1, 1, "ss_3e-4",        1, 4, 2050808, -1, "PPO", None),    # 3

    # 4-6 (PPO for lr=1e-4)
    (deathmatch_bot, 1e-4,      1, 1, "ss_rgb_1e-4",    2, 4, 2050808, -1, "PPO", None),    # 4
    (deathmatch_bot, 1e-4,      1, 1, "rgb_1e-4",       0, 4, 2050808, -1, "PPO", None),    # 5
    (deathmatch_bot, 1e-4,      1, 1, "ss_1e-4",        1, 4, 2050808, -1, "PPO", None),    # 6

    # 7-9 (PPO for lr=9e-4)
    (deathmatch_bot, 9e-4,      1, 1, "ss_rgb_9e-4",    2, 4, 2050808, -1, "PPO", None),    # 7
    (deathmatch_bot, 9e-4,      1, 1, "rgb_9e-4",       0, 4, 2050808, -1, "PPO", None),    # 8
    (deathmatch_bot, 9e-4,      1, 1, "ss_9e-4",        1, 4, 2050808, -1, "PPO", None),    # 9

    # 10-11 (PPO for ss, lr = [1e-3, 1e-5])
    (deathmatch_bot, 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808, -1, "PPO", None),    # 10
    (deathmatch_bot, 1e-5,      1, 1, "ss_1e-5",        1, 4, 2050808, -1, "PPO", None),    # 11
    
    # 12-13 (PPO for rgb, lr = [1e-3, 1e-5])
    (deathmatch_bot, 1e-3,      1, 1, "rgb_1e-3",       0, 4, 2050808, -1, "PPO", None),    # 12
    (deathmatch_bot, 1e-5,      1, 1, "rgb_1e-5",       0, 4, 2050808, -1, "PPO", None),    # 13

    # 14-15 (PPO for ss+rgb, lr = [1e-3, 1e-5])
    (deathmatch_bot, 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808, -1, "PPO", None),    # 14
    (deathmatch_bot, 1e-5,      1, 1, "ss_rgb_1e-5",    2, 4, 2050808, -1, "PPO", None),    # 15

    # 16-18 (IQN for lr=1e-3, with framestack)
    (deathmatch_bot, 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808,  1, "IQN", "logs/stack_ppo_ss_rgb_1e-3/best_model.zip"), # 16
    (deathmatch_bot, 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808,  1, "IQN", "logs/stack_ppo_ss_1e-3/best_model.zip"),     # 17
    (deathmatch_bot, 1e-3,      1, 1, "rgb_1e-3",       0, 4, 2050808,  1, "IQN", "logs/stack_ppo_rgb_9e-3/best_model.zip"),    # 18
    
    # 19-21 (IQN for lr=1e-3, without framestack)
    (deathmatch_bot, 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808,  0, "IQN", "logs/ss_rgb_1e-3/best_model.zip"),   # 19
    (deathmatch_bot, 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808,  0, "IQN", "logs/ss_1e-3/best_model.zip"),       # 20
    (deathmatch_bot, 1e-3,      1, 1, "rgb_1e-3",       0, 4, 2050808,  0, "IQN", "logs/rgb_9e-3/best_model.zip"),      # 21
    
][task_idx:task_idx+1]

tasks = [(deathmatch_bot, 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808,  0, "IQN", "logs/ss_1e-3/best_model.zip")]

def check_gpu() -> torch.device:
    """Checks the system to see if CUDA devices are available.
    Warning:
    cudnn.benchmark hurts performance if convolutional layers receives varying input shape

    Returns:
        torch.device: device for running PyTorch with
    """    
    # Uses GPU if available
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print("\nCUDA device detected.")
    else:
        DEVICE = torch.device("cpu")
        print("\nNo CUDA device detected.")
    return DEVICE

def main():
    global tasks
    # global save_validation, agent
    # DEVICE = check_gpu()

    # ep_max (maximum training episode) would overwrite the epoch_num (number of epochs) setting
    for config, lr, ep_max, save_interval, name, input_rep, n_envs, seed, framestack, model_str, warmup_model in tasks:
        model_str = model_str.casefold()
        name = f"{model_str}_{name}"
        if model_str[:2] == "r_":
            framestack = 0
        if framestack < 0:
            framestack = global_framestack
        ch_num = input_rep_ch_num[input_rep]
        if framestack:
            ch_num *= frame_repeat
        game_config = dict(config_path=config, color=True, label=True, res=(256, 144), visibility=False, add_args=bot_args)

        game = create_game(**game_config)
        n = game.get_available_buttons_size()
        game.close()
        del game

        act_actions = (tuple(x) for x in it.product([False, True], repeat=n) if any(x))
        act_actions = (x for x in act_actions if not (x[2] and x[3]))   # filter out strafe left and right simultaneously
        act_actions = (x for x in act_actions if not (x[4] and x[5]))   # filter out turn left and right simultaneously
        act_actions = [x for x in act_actions if not x[0]]              # filter out attack moves
        act_actions = [tuple([True] + [False] * (n-1))] + act_actions   # add back attack move when standing still

        env_kwargs = {
            "frame_stack"   : bool(framestack),
            "realtime_ss"   : "rtss" in name,
            "actions"       : act_actions, 
            "input_shape"   : (ch_num, 144, 256), 
            "game_config"   : game_config,
            "frame_repeat"  : frame_repeat,
            "seed"          : seed,
            "input_rep"     : input_rep
        }

        policy_kwargs = {
            "normalize_images"          : True,
            "features_extractor_class"  : CustomCNN,
            "features_extractor_kwargs" : {"features_dim" : 128}
        }

        if input_rep == 3:
            policy_kwargs["normalize_images"] = False

        t = time()
        env_type = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        env = make_vec_env(DoomBotDeathMatch, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=env_type)
        eval_env_kwargs = env_kwargs.copy()
        eval_game_config = game_config.copy()
        eval_game_config["add_args"] = bot_args_eval
        eval_env_kwargs["is_eval"] = True
        eval_env_kwargs["game_config"] = eval_game_config
        eval_env = make_vec_env(DoomBotDeathMatch, n_envs=2 if n_envs > 1 else 1, seed=seed, env_kwargs=eval_env_kwargs, vec_env_cls=env_type)

        if framestack:
            name = f"stack_{name}"

        eval_callback =  EvalCallbackWithWebhook(
            eval_env,
            n_eval_episodes=10,
            best_model_save_path=f'./logs/{name}', 
            log_path=f'./logs/{name}', 
            eval_freq=4096 * 2,
            deterministic=True, 
            render=False
        )

        if n_envs == 1:
            print(f"Took {time() - t:.2f} seconds to generate training/evaluation environments.")

        t = time()
        match model_str.casefold():
            case "ppo":
                model = PPO("CnnPolicy", env, learning_rate=lr, verbose=1, n_steps=4096, batch_size=32, 
                            n_epochs=3, policy_kwargs=policy_kwargs, seed=seed)
                model_desc = ""
            case "r_ppo":
                model = RecurrentPPO("CnnLstmPolicy", env, learning_rate=lr, verbose=1, n_steps=4096, 
                                     batch_size=32, n_epochs=3, policy_kwargs=policy_kwargs, seed=seed)
                model_desc = ""
            case "iqn":
                IQN_policy_kwargs = {
                    # "n_quantiles": 32,
                    # "num_cosine": 32,
                }
                IQN_policy_kwargs.update(policy_kwargs)

                # optimize_memory_usage unsupported for IQN
                model = IQN("CnnPolicy", env, learning_rate=lr, verbose=1, learning_starts=0, buffer_size=409_600,
                            # replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(goal_selection_strategy="episode"), 
                            train_freq=(4096, "step"), gradient_steps=3, batch_size=32, target_update_interval=4096,
                            optimize_memory_usage=False, policy_kwargs=IQN_policy_kwargs, seed=seed, exploration_initial_eps=1,
                            exploration_final_eps=0.01, exploration_fraction=0.025)

                model_desc = f"\nbuffer size: {model.buffer_size}\nwarmup: {warmup_model}"

                if isinstance(warmup_model, str) and os.path.exists(warmup_model):
                    assert warmup_n_envs % n_envs == 0, f"warmup_n_envs ({warmup_n_envs}) incompatible with n_envs ({n_envs})"
                    
                    env = make_vec_env(DoomBotDeathMatch, n_envs=warmup_n_envs, seed=seed, env_kwargs=eval_env_kwargs, vec_env_cls=env_type)
                    ppo_for_warmup = PPO.load(warmup_model)
                    env_iter = list(range(warmup_n_envs))
                    finished_eps_count = 0

                    pbar = tqdm(total=warmup_episodes, desc=f"Warming up ({warmup_episodes})")

                    done = np.zeros(warmup_n_envs, dtype='?')
                    obs = env.reset()
                    reward_sum = 0
                    env_total_reward = np.zeros(warmup_n_envs, dtype=np.float64)

                    reshape_factor = warmup_n_envs // n_envs
                    reshape_iter = list(range(reshape_factor))

                    while finished_eps_count < warmup_episodes:
                        action, _ = ppo_for_warmup.predict(obs, deterministic=True)
                        new_obs, reward, done, info = env.step(action)
                        env_total_reward += reward
                        if done.any():
                            reward_sum += env_total_reward[done].sum()
                            env_total_reward[done] = 0
                            new_finished_episodes = done.sum()
                            finished_eps_count += new_finished_episodes
                            pbar.update(new_finished_episodes)
                        for i in reshape_iter:
                            j = i * n_envs
                            k = j + n_envs
                            model.replay_buffer.add(obs[j:k,...], new_obs[j:k,...], action[j:k], reward[j:k], done[j:k], info[j:k])
                        obs = new_obs
                    model_desc = model_desc + f" (mean reward: {reward_sum / finished_eps_count:.2f})"
                    pbar.set_description_str("Closing environments")
                    env.close()
                    pbar.set_description_str("Cleaning CUDA cache")
                    torch.cuda.empty_cache()
                    pbar.set_description_str("Garbage collection")
                    del env, ppo_for_warmup, env_iter, obs, new_obs, reward, done, info, action
                    gc.collect()
                    pbar.close()
        
        if n_envs == 1:
            print(f"Took {time() - t:.2f} seconds to create agent.")
        
        print(f"Number of available buttons: {n} | Size of action space: {len(act_actions)}")
        msg = f"Model: {model_str.upper()}, Framestack: {bool(framestack)}{model_desc}"
        print(msg, flush=True)
        print(model.policy, flush=True)

        # webhook = discord_bot(extra=name)
        # eval_callback.attach_webhook(webhook, name=name)
        # webhook.send_msg(msg)

        model.learn(total_timesteps=4_000_000, callback=eval_callback, progress_bar=True)
        model.save(f"./logs/{name}/final.zip")

if __name__ == "__main__":
    main()
