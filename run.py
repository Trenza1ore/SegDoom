import os
import gc
import itertools as it
from time import sleep, time

import vizdoom as vzd
from tqdm.rich import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, Adagrad

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO as RPPO, IQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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
global_framestack = 0
global_warmup_episodes = 200
warmup_n_envs = 40
frame_repeat = 4

input_rep_ch_num = {
    0   : 3,    # RGB
    1   : 1,    # Semantic Segmentation Mask
    2   : 4,    # RGBS (S = Semantic Segmentation Mask)
    3   : 3,    # Semantic Segmentation as RGB
}

iqn_1 = dict(
    train_freq              = (4096, "step"),
    gradient_steps          = 3,
    target_update_interval  = 4096,
)

iqn_2 = dict(
    train_freq              = (128, "step"),
    gradient_steps          = 3,
    target_update_interval  = 4096,
)

iqn_3 = dict(
    train_freq              = (4, "step"),
    gradient_steps          = 1,
    target_update_interval  = 2048,
)

rppo = dict(env_config={"n_updates" : 1})

rps = dict(env_config={"n_updates" : 1}, n_steps=2048, batch_size=64, 
           policy_config={"lstm_hidden_size" : 64, "n_lstm_layers" : 1})

rpm = rps.copy()
rpm.update(dict(env_config={"n_updates" : 1}, n_epochs=4, 
                policy_config={"lstm_hidden_size" : 128, "n_lstm_layers" : 1}))

st4 = dict(env_config={"n_updates" : 1, "frame_repeat" : 4}, batch_size=64, 
           policy_config={"net_arch" : dict(pi=[128, 128], vf=[128, 128])})

# config, lr, ep_max, save_interval, name, input_rep, n_envs, seed, framestack, model_str, teacher, variables
# framestack: -1 = global behavior, 0 = Off, 1 = On
tasks = [
    # 1-3 (PPO for lr=3e-4)
    (deathmatch_bot, 3e-4, "ss_rgb_3e-4",    2, 4, 2050808, -1, "PPO", None, {}),    # 1
    (deathmatch_bot, 3e-4, "rgb_3e-4",       0, 4, 2050808, -1, "PPO", None, {}),    # 2
    (deathmatch_bot, 3e-4, "ss_3e-4",        1, 4, 2050808, -1, "PPO", None, {}),    # 3

    # 4-6 (PPO for lr=1e-4)
    (deathmatch_bot, 1e-4, "ss_rgb_1e-4",    2, 4, 2050808, -1, "PPO", None, {}),    # 4
    (deathmatch_bot, 1e-4, "rgb_1e-4",       0, 4, 2050808, -1, "PPO", None, {}),    # 5
    (deathmatch_bot, 1e-4, "ss_1e-4",        1, 4, 2050808, -1, "PPO", None, {}),    # 6

    # 7-9 (PPO for lr=9e-4)
    (deathmatch_bot, 9e-4, "ss_rgb_9e-4",    2, 4, 2050808, -1, "PPO", None, {}),    # 7
    (deathmatch_bot, 9e-4, "rgb_9e-4",       0, 4, 2050808, -1, "PPO", None, {}),    # 8
    (deathmatch_bot, 9e-4, "ss_9e-4",        1, 4, 2050808, -1, "PPO", None, {}),    # 9

    # 10-11 (PPO for ss, lr = [1e-3, 1e-5])
    (deathmatch_bot, 1e-3, "ss_1e-3",        1, 4, 2050808, -1, "PPO", None, {}),    # 10
    (deathmatch_bot, 1e-5, "ss_1e-5",        1, 4, 2050808, -1, "PPO", None, {}),    # 11
    
    # 12-13 (PPO for rgb, lr = [1e-3, 1e-5])
    (deathmatch_bot, 1e-3, "rgb_1e-3",       0, 4, 2050808, -1, "PPO", None, {}),    # 12
    (deathmatch_bot, 1e-5, "rgb_1e-5",       0, 4, 2050808, -1, "PPO", None, {}),    # 13

    # 14-15 (PPO for ss+rgb, lr = [1e-3, 1e-5])
    (deathmatch_bot, 1e-3, "ss_rgb_1e-3",    2, 4, 2050808, -1, "PPO", None, {}),    # 14
    (deathmatch_bot, 1e-5, "ss_rgb_1e-5",    2, 4, 2050808, -1, "PPO", None, {}),    # 15

    # [Failures]
    # 16-18 (IQN for lr=1e-3, with framestack)
    (deathmatch_bot, 1e-3, "ss_rgb_1e-3",    2, 4, 2050808,  4, "IQN",               # 16
     "logs/stack_ppo_ss_rgb_1e-3/best_model.zip", iqn_1), 
    (deathmatch_bot, 1e-3, "ss_1e-3",        1, 4, 2050808,  4, "IQN",               # 17
     "logs/stack_ppo_ss_1e-3/best_model.zip", iqn_1),     # 17
    (deathmatch_bot, 1e-3, "rgb_1e-3",       0, 4, 2050808,  4, "IQN",               # 18
     "logs/stack_ppo_rgb_9e-3/best_model.zip", iqn_1),    # 18
    
    # [Failures]
    # 19-21 (IQN for lr=1e-3, without framestack) 4096 train_freq, 3 grad steps, 4096 target update freq
    (deathmatch_bot, 1e-3, "ss_rgb_1e-3",    2, 4, 2050808,  0, "IQN",               # 19
     "logs/ss_rgb_1e-3/best_model.zip", iqn_1),
    (deathmatch_bot, 1e-3, "ss_1e-3",        1, 4, 2050808,  0, "IQN",               # 20
     "logs/ss_rgb_1e-3/best_model.zip", iqn_1),
    (deathmatch_bot, 1e-3, "rgb_1e-3",       0, 4, 2050808,  0, "IQN",               # 21
     "logs/ss_rgb_1e-3/best_model.zip", iqn_1),

    # [Failures]
    # 22 128 train_freq, 3 grad steps, 4096 target update freq
    (deathmatch_bot, 1e-3, "128_ss_1e-3",    1, 4, 2050808,  0, "IQN",               # 22
     "logs/ss_rgb_1e-3/best_model.zip", {}),

    # [Failures]
    # 22 4 train_freq, 1 grad steps, 2048 target update freq
    (deathmatch_bot, 1e-3, "4_ss_1e-3",      1, 4, 2050808,  0, "IQN",               # 23
     "logs/ss_rgb_1e-3/best_model.zip", {}),
    
    # [Failures]
    # 24-26 (Recurrent PPO for ss+rgb and ss, lr=1e-3)
    (deathmatch_bot, 1e-3, "ss_rgb_1e-3",    2, 4, 2050808,  2, "R_PPO", None, rppo),# 24
    (deathmatch_bot, 1e-3, "ss_1e-3",        1, 4, 2050808,  8, "R_PPO", None, rppo),# 25
    (deathmatch_bot, 1e-3, "rgb_1e-3",       0, 4, 2050808,  8, "R_PPO", None, rppo),# 26
    (deathmatch_bot, 1e-3, "ss_rgb_1e-3",    2, 4, 2050808,  8, "R_PPO", None, rppo),# 27
    
    # 28-29 Recurrent PPOs (high lr): small model with lr=1e-3 and medium model with lr=5e-4
    (deathmatch_bot, 1e-3, "sm_ss_1e-3",     1, 4, 2050808,  1, "R_PPO", None,  rps),# 28
    (deathmatch_bot, 5e-4, "md_ss_5e-4",     1, 4, 2050808,  3, "R_PPO", None,  rpm),# 29

    # 30-31 PPO with framestacking (lr=1e-3)
    (deathmatch_bot, 1e-3, "ss_1e-3",        1, 4, 2050808,  4,   "PPO", None,  st4),# 30
    (deathmatch_bot, 5e-4, "ss_5e-4",        1, 4, 2050808,  4,   "PPO", None,  st4),# 31

    # 32-33 Recurrent PPOs ( mid lr): small model with lr=5e-4 and medium model with lr=3e-4
    (deathmatch_bot, 5e-4, "sm_ss_5e-4",     1, 4, 2050808,  1, "R_PPO", None,  rps),# 32
    (deathmatch_bot, 3e-4, "md_ss_3e-4",     1, 4, 2050808,  3, "R_PPO", None,  rpm),# 33

    # 34-35 Recurrent PPOs ( low lr): small model with lr=3e-4 and medium model with lr=1e-4
    (deathmatch_bot, 3e-4, "sm_ss_3e-4",     1, 4, 2050808,  1, "R_PPO", None,  rps),# 34
    (deathmatch_bot, 1e-4, "md_ss_1e-4",     1, 4, 2050808,  3, "R_PPO", None,  rpm),# 35

    # 36-37 PPO with framestacking (lr=1e-5)
    (deathmatch_bot, 1e-3, "ss_rgb_1e-3",    2, 4, 2050808,  4,   "PPO", None,  st4),# 36
    (deathmatch_bot, 5e-4, "ss_rgb_5e-4",    2, 4, 2050808,  4,   "PPO", None,  st4),# 37
    
][task_idx:task_idx+1]

# tasks = [(deathmatch_bot, 1e-3, "ss_1e-3",        1, 4, 2050808,  0, "IQN", "logs/ss_1e-3/best_model.zip")]

def main():
    global tasks
    # global save_validation, agent
    # DEVICE = check_gpu()

    # ep_max (maximum training episode) would overwrite the epoch_num (number of epochs) setting
    for config, lr, name, input_rep, n_envs, seed, framestack, model_str, warmup_model, model_config in tasks:
        model_config: dict[str, object]

        model_str = model_str.casefold()
        name = f"{model_str}_{name}"

        if framestack < 0:
            framestack = global_framestack
        ch_num = input_rep_ch_num[input_rep]
        if framestack:
            ch_num *= framestack
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
            "buffer_size"   : int(framestack),
            "realtime_ss"   : "rtss" in name,
            "actions"       : act_actions, 
            "input_shape"   : (ch_num, 144, 256), 
            "game_config"   : game_config,
            "frame_repeat"  : frame_repeat,
            "seed"          : seed,
            "input_rep"     : input_rep
        }

        if "env_config" in model_config:
            env_kwargs.update(model_config["env_config"])
            del model_config["env_config"]

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
        eval_env = make_vec_env(DoomBotDeathMatch, n_envs=10, seed=seed, env_kwargs=eval_env_kwargs, vec_env_cls=env_type)

        if framestack:
            if "r_" in model_str:
                name = f"r{framestack}_{name}"
            else:
                name = f"s{framestack}_{name}"

        eval_callback =  EvalCallbackWithWebhook(
            eval_env,
            n_eval_episodes=10,
            best_model_save_path=f'./logs/{name}', 
            log_path=f'./logs/{name}', 
            eval_freq=4096*2,
            deterministic=True, 
            render=False
        )

        if n_envs == 1:
            print(f"Took {time() - t:.2f} seconds to generate training/evaluation environments.")

        mem_available = -1
        try:
            import psutil
        except:
            psutil = None

        t = time()
        model_desc = ""

        match model_str.casefold():
            case "ppo":
                PPO_policy_kwargs = policy_kwargs.copy()
                PPO_policy_kwargs.update({

                })
                PPO_policy_kwargs.update(model_config.get("policy_config", {}))

                n_steps = model_config.get("n_steps", 4096)
                n_epochs =  model_config.get("n_epochs", 3)
                batch_size =  model_config.get("batch_size", 32)

                model = PPO("CnnPolicy", env, learning_rate=lr, verbose=1, n_steps=n_steps, batch_size=batch_size, 
                            n_epochs=n_epochs, policy_kwargs=PPO_policy_kwargs, seed=seed)
                if psutil is not None:
                    model_desc = f"\nMemory available: {psutil.virtual_memory().available / 1e9:.2f} GB"
            case "r_ppo":
                RPPO_policy_kwargs = policy_kwargs.copy()
                RPPO_policy_kwargs.update({
                    "lstm_hidden_size"  : 128,
                    "n_lstm_layers"     : 2,
                })
                RPPO_policy_kwargs.update(model_config.get("policy_config", {}))

                n_steps = model_config.get("n_steps", 4096)
                n_epochs =  model_config.get("n_epochs", 3)
                batch_size =  model_config.get("batch_size", 32)

                model = RPPO("CnnLstmPolicy", env, learning_rate=lr, verbose=1, n_steps=n_steps,
                                     batch_size=batch_size, n_epochs=n_epochs, policy_kwargs=RPPO_policy_kwargs, seed=seed)
                if psutil is not None:
                    model_desc = f"\nMemory available: {psutil.virtual_memory().available / 1e9:.2f} GB"
            case "iqn":
                IQN_policy_kwargs = {
                    # "n_quantiles": 32,
                    # "num_cosine": 32,
                    # "net_arch": [64,64]
                }
                IQN_policy_kwargs.update(policy_kwargs)

                try:
                    import psutil
                    mem_available = psutil.virtual_memory().available / 1e9
                except:
                    pass
                
                exp_frac = 0.1
                model = IQN("CnnPolicy", env, learning_rate=lr, verbose=1, learning_starts=0, buffer_size=409600//ch_num,
                            replay_buffer_class=ReplayBuffer, replay_buffer_kwargs=dict(handle_timeout_termination = False),  # ER
                            policy_kwargs=IQN_policy_kwargs, optimize_memory_usage=True, batch_size=32, tau=0.01, seed=seed, 
                            exploration_initial_eps=1, exploration_final_eps=0.01, exploration_fraction=exp_frac, **model_config)
                
                total_memory_usage = model.replay_buffer.observations.nbytes + model.replay_buffer.actions.nbytes
                total_memory_usage += model.replay_buffer.rewards.nbytes + model.replay_buffer.dones.nbytes
                mem_usage = f"buffer size: {model.buffer_size}\nbuffer memory usage: {total_memory_usage/1e9:.2f} GB / {mem_available:.2f} GB available"
                model_desc = f"\n{mem_usage}\nwarmup: {warmup_model}"

                if isinstance(warmup_model, str) and os.path.exists(warmup_model):
                    global warmup_n_envs
                    
                    warmup_episodes = min(round(model.replay_buffer.buffer_size * n_envs * 0.9 / 1300), global_warmup_episodes)

                    if warmup_n_envs > warmup_episodes:
                        warmup_n_envs = (warmup_episodes // n_envs) * n_envs
                    
                    assert warmup_n_envs % n_envs == 0 and warmup_n_envs >= n_envs, f"warmup_n_envs value ({warmup_n_envs}) incompatible with n_envs ({n_envs})"

                    warmup_env_kwargs = eval_env_kwargs.copy()
                    warmup_input_rep = 2
                    warmup_env_kwargs["input_shape"] = (input_rep_ch_num[warmup_input_rep], 144, 256)
                    warmup_env_kwargs["input_rep"] = warmup_input_rep
                    env = make_vec_env(DoomBotDeathMatch, n_envs=warmup_n_envs, seed=seed, env_kwargs=warmup_env_kwargs, vec_env_cls=env_type)

                    ppo_for_warmup = PPO.load(warmup_model)
                    env_iter = list(range(warmup_n_envs))
                    finished_eps_count = 0
                    
                    import sys
                    if sys.platform.lower()[:3] == "win":
                        os.system("cls")
                    else:
                        os.system("clear")
                    print(mem_usage, flush=True)

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
                            model.replay_buffer.add(obs[j:k,3:4,...], new_obs[j:k,3:4,...], action[j:k], reward[j:k], done[j:k], info[j:k])
                        obs = new_obs
                    model_desc = model_desc + f" (mean reward: {reward_sum / finished_eps_count:.2f})"
                    pbar.set_description_str("Closing environments")
                    env.close()
                    pbar.set_description_str("Cleaning CUDA cache")
                    torch.cuda.empty_cache()
                    pbar.set_description_str("Garbage collection")
                    del env, ppo_for_warmup, env_iter, obs, new_obs, reward, done, info, action, env_total_reward
                    gc.collect()
                    pbar.close()
                else:
                    import sys
                    if sys.platform.lower()[:3] == "win":
                        os.system("cls")
                    else:
                        os.system("clear")
                    print(mem_usage, flush=True)
        
        if n_envs == 1:
            print(f"Took {time() - t:.2f} seconds to create agent.")
        
        msg = f"Number of available buttons: {n} | Size of action space: {len(act_actions)} | "
        msg += f"Input shape: {env_kwargs['input_shape']}\n"
        msg += f"Model: {model_str.upper()}, Framestack: {bool(framestack)}{model_desc}"
        print(model.policy, flush=True)
        print(msg, flush=True)

        webhook = DiscordWebhook(extra=name)
        eval_callback.attach_webhook(webhook, name=name)
        webhook.send_msg(msg)

        model.learn(total_timesteps=4_000_000, callback=eval_callback, progress_bar=True)
        model.save(f"./logs/{name}/final.zip")
        model.env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
