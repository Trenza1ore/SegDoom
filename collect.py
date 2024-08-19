import os
import sys
import itertools as it
from time import sleep, time

import psutil
from tqdm.rich import tqdm

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO, IQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from models import *
from wrapper import *
from vizdoom_utils import *
from models.training_procedure import *

# ============================== What is this ========================================
# The program to collect gameplay footage/data/performance of evaluation sessions
# ====================================================================================

task_idx = 0                # set this to -1 to execute all tasks in tasks list

# config
save_dir = "./logs"
eps_to_eval = 200
save_batch_size = 200       # (roughly) how many episodes are saved simultaneously, no promise
global_n_env_eval = 40      # number of venv (vectorized environment) to use by default
global_n_env_eval_rtss = 4  # number of venv to use for real-time semantic segmentation
env_type = SubprocVecEnv    # type of venv, SubprocVecEnv for multi-processing, DummyVecEnv 
# env_type = DummyVecEnv    # DummyVecEnv isn't fully supported

# scenarios_path = vzd.scenarios_path
scenarios_path = "./scenarios"

# Deathmatch Scenarios
maps = {
    "map1"  : os.path.join(scenarios_path, "bots_deathmatch_1.cfg"),
    "map1a" : os.path.join(scenarios_path, "bots_deathmatch_1a.cfg"),
    "map1w" : os.path.join(scenarios_path, "bots_deathmatch_1w.cfg"),
    "map2"  : os.path.join(scenarios_path, "bots_deathmatch_2.cfg"),
    "map2s" : os.path.join(scenarios_path, "bots_deathmatch_2s.cfg"),
    "map3"  : os.path.join(scenarios_path, "bots_deathmatch_3.cfg"),
}
for k in list(maps.keys()):
    if "rtss_" not in k:
        maps["rtss_" + k] = maps[k]

# Game arguments
bot_args = bot_args_eval = " ".join([
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

# Definition for model input's representations
input_rep_ch_num = {
    0   : 3,    # type 0: 3-channel RGB , coloured game frame
    1   : 1,    # type 1: 1-channel S   , ss (semantic segmentation) mask
    2   : 4,    # type 2: 4-channel RGBS, coloured game frame + ss mask
    3   : 3     # type 3: 3-channel RGB , ss mask mapped to RGB via colour map
}

# Tasks to perform
tasks = [
    ("map1"         , "ss_rgb_3e-4",        2, 2050808, {}, ''),       # 1
    ("map1"         , "rgb_3e-4",           0, 2050808, {}, ''),       # 2
    ("map1"         , "ss_3e-4",            1, 2050808, {}, ''),       # 3

    ("map1"         , "ss_rgb_1e-4",        2, 2050808, {}, ''),       # 4
    ("map1"         , "rgb_1e-4",           0, 2050808, {}, ''),       # 5
    ("map1"         , "ss_1e-4",            1, 2050808, {}, ''),       # 6

    ("map1"         , "ss_rgb_9e-4",        2, 2050808, {}, ''),       # 7
    ("map1"         , "rgb_9e-4",           0, 2050808, {}, ''),       # 8
    ("map1"         , "ss_9e-4",            1, 2050808, {}, ''),       # 9

    ("map1"         , "ss_1e-3",            1, 2050808, {}, ''),       # 10
    ("map1"         , "ss_1e-5",            1, 2050808, {}, ''),       # 11

    ("map2s"        , "ss_1e-3",            1, 2050808, {}, ''),       # 12
    ("map2s"        , "ss_1e-5",            1, 2050808, {}, ''),       # 13

    ("map3"         , "ss_1e-3",            1, 2050808, {}, ''),       # 14
    ("map3"         , "ss_1e-5",            1, 2050808, {}, ''),       # 15

    ("rtss_map1"    , "ss_1e-3",            1, 2050808, {}, ''),       # 16
    ("rtss_map1"    , "ss_1e-5",            1, 2050808, {}, ''),       # 17

    ("rtss_map2s"   , "ss_1e-3",            1, 2050808, {}, ''),       # 18
    ("rtss_map2s"   , "ss_1e-5",            1, 2050808, {}, ''),       # 19

    ("rtss_map3"    , "ss_1e-3",            1, 2050808, {}, ''),       # 20
    ("rtss_map3"    , "ss_1e-5",            1, 2050808, {}, ''),       # 21

    ("map1"         , "ss_rgb_1e-3",        2, 2050808, {}, ''),       # 22
    ("rtss_map1"    , "ss_rgb_1e-3",        2, 2050808, {}, ''),       # 23
    ("map2s"        , "ss_rgb_1e-3",        2, 2050808, {}, ''),       # 24
    ("rtss_map2s"   , "ss_rgb_1e-3",        2, 2050808, {}, ''),       # 25
    ("map3"         , "ss_rgb_1e-3",        2, 2050808, {}, ''),       # 26
    ("rtss_map3"    , "ss_rgb_1e-3",        2, 2050808, {}, ''),       # 27

    ("map1"         , "stack_ppo_ss_1e-3",  1, 2050808, {}, ''),       # 28
    ("rtss_map1"    , "stack_ppo_ss_1e-3",  1, 2050808, {}, ''),       # 29
    ("map2s"        , "stack_ppo_ss_1e-3",  1, 2050808, {}, ''),       # 30
    ("rtss_map2s"   , "stack_ppo_ss_1e-3",  1, 2050808, {}, ''),       # 31
    ("map3"         , "stack_ppo_ss_1e-3",  1, 2050808, {}, ''),       # 32
    ("rtss_map3"    , "stack_ppo_ss_1e-3",  1, 2050808, {}, ''),       # 33
    
    ("map1"         , "stack_ppo_ss_1e-3",  1, 2050808, {}, 'final'),  # 34
    ("rtss_map1"    , "stack_ppo_ss_1e-3",  1, 2050808, {}, 'final'),  # 35
    ("map2s"        , "stack_ppo_ss_1e-3",  1, 2050808, {}, 'final'),  # 36
    ("rtss_map2s"   , "stack_ppo_ss_1e-3",  1, 2050808, {}, 'final'),  # 37
    ("map3"         , "stack_ppo_ss_1e-3",  1, 2050808, {}, 'final'),  # 38
    ("rtss_map3"    , "stack_ppo_ss_1e-3",  1, 2050808, {}, 'final'),  # 39
]

if task_idx >= 0:
    tasks = tasks[task_idx:task_idx+1]

# tasks = [("rtss_map2s"   , "ss_rgb_1e-3",       2, 2050808)]

env_kwargs_template = {
    "smooth_frame"  : True,
    "frame_repeat"  : 4,
    "only_pos"      : True,
}

def main():
    global tasks
    
    for config, name, input_rep, seed, additonal_config, model_path in tasks:
        save_name = f"{config}_{name}"
        ch_num = input_rep_ch_num[input_rep]
        game_config = dict(config_path=maps[config], color=True, label=True, res=(256, 144), visibility=False, add_args=bot_args)

        game = create_game(**game_config)
        n = game.get_available_buttons_size()
        act_actions = (tuple(x) for x in it.product([False, True], repeat=n) if any(x))
        act_actions = (x for x in act_actions if not (x[2] and x[3]))   # filter out strafe left (3) and right (4) simultaneously
        act_actions = (x for x in act_actions if not (x[4] and x[5]))   # filter out turn   left (4) and right (5) simultaneously
        act_actions = [x for x in act_actions if not (x[0]         )]   # filter out actions with attack (0)
        act_actions = [tuple([True] + [False] * (n-1))] + act_actions   # add back one action that performs attack (0) only

        game.close()
        del game

        p = tqdm(total=eps_to_eval)
        eval_callback = lambda locals_dict, _: (scores.append(locals_dict['info']['r']), p.update()) if locals_dict['done'] else None

        n_env_eval = global_n_env_eval_rtss if "rtss" in save_name else global_n_env_eval
        
        mem_size_est = additonal_config.get("memsize", None)
        if mem_size_est is None:
            mem_size_est = round(1300 * eps_to_eval / n_env_eval)
            if n_env_eval > 1:
                mem_size_est += max(round(mem_size_est / 10), 2600)

        env_kwargs = env_kwargs_template.copy()
        env_kwargs.update({
            "realtime_ss"   : "rtss" in save_name,
            "frame_stack"   : "stack" in save_name,
            "actions"       : act_actions, 
            "input_shape"   : (ch_num, 144, 256), 
            "game_config"   : game_config,
            "seed"          : seed,
            "input_rep"     : input_rep,
            "memsize"       : mem_size_est,
        })

        policy_kwargs = {
            "normalize_images"          : True,
            "features_extractor_class"  : CustomCNN,
            "features_extractor_kwargs" : {"features_dim" : 128}
        }

        if input_rep == 3:
            policy_kwargs["normalize_images"] = False

        t = time()
        eval_env_kwargs = env_kwargs.copy()
        eval_game_config = game_config.copy()
        eval_game_config["add_args"] = bot_args_eval
        eval_env_kwargs["game_config"] = eval_game_config
        eval_env: SubprocVecEnv | DummyVecEnv | VecEnv = make_vec_env(
            DoomBotDeathMatchCapture, n_envs=n_env_eval, seed=seed, env_kwargs=eval_env_kwargs, vec_env_cls=env_type)

        if model_path:
            if not os.path.exists(model_path):
                model_path = f"{model_path}.zip"
        else:
            model_path = "best_model.zip"


        mem = psutil.virtual_memory().available / 1e9
        # Not the most elegant way to do it, but if it works, it works...
        try:
            model = PPO.load(f"{save_dir}/{name}/{model_path}")
        except:
            try:
                model = RecurrentPPO.load(f"{save_dir}/{name}/{model_path}")
            except:
                model = IQN.load(f"{save_dir}/{name}/{model_path}")

        if sys.platform.lower()[:3] == "win":
            os.system("cls")
        else:
            os.system("clear")
            
        mem_after_model_alloc = psutil.virtual_memory().available / 1e9
        prob_alloc = mem - mem_after_model_alloc
        print(f"{mem_after_model_alloc:.2f} GB available, model probably occupied {prob_alloc:.2f} GB", flush=True)

        print(f"Number of available buttons: {n} | Size of action space: {len(act_actions)}", flush=True)
        p.update(0)
        scores = []
        evaluate_policy(model, eval_env, eps_to_eval, callback=eval_callback)
        p.set_description("Finished evaluations")
        p.close()

        if not os.path.exists(f"{save_dir}/{save_name}"):
            os.makedirs(f"{save_dir}/{save_name}")

        scores = np.array(scores, dtype=np.int64)
        np.save(f"{save_dir}/{save_name}/performance.npy", scores)
        print(f"Results saved to {save_dir}/{save_name}/performance.npy", flush=True)
        print(f"Scores: {scores.tolist()}\n{scores.mean():.2f} +/- {scores.std():.3f} | min: {np.min(scores):2d} | Q1: {int(np.percentile(scores, 25)):2d} | Q2: {int(np.percentile(scores, 50)):2d} | Q3: {int(np.percentile(scores, 75)):2d} | max: {np.max(scores):2d}")

        fps_records = []

        # Use duck typing to our advantage, that's why I love OOP :-)
        if isinstance(eval_env, SubprocVecEnv):
            eval_env: SubprocVecEnv
            f_get_envs = eval_env._get_target_remotes
            f_send_save_instruction = lambda i, remote: remote.send(
                ("env_method", ("save", [f"{save_dir}/{save_name}/record_{i}.npz"], {}))
            )
            f_receive_results = lambda remotes: [remote.recv() for remote in remotes]
        elif isinstance(eval_env, DummyVecEnv):
            eval_env: DummyVecEnv
            f_get_envs = eval_env._get_target_envs
            f_send_save_instruction = lambda i, remote: fps_records.append(
                getattr(remote, "save")(f"{save_dir}/{save_name}/record_{i}.npz")
            )
            f_receive_results = lambda _: list()
        else:
            raise NotImplementedError(f"eval_env has an unknown type: {type(eval_env)}")
        
        # Praise duck typing, calls "save" method in our custom gymnasium wrapper
        task_queue = []
        target_remotes = f_get_envs(None)
        if eps_to_eval <= save_batch_size or env_kwargs.get("only_pos", False):
            for i, remote in enumerate(target_remotes):
                f_send_save_instruction(i, remote)
            fps_records += f_receive_results(target_remotes)
        else:
            task_queue = []
            queue_limit = (eps_to_eval // n_env_eval) // save_batch_size
            for i, remote in enumerate(target_remotes):
                f_send_save_instruction(i, remote)
                task_queue.append(remote)
                if len(task_queue) >= queue_limit:
                    fps_records += f_receive_results(task_queue)
                    task_queue.clear()
        
        if len(task_queue):
            fps_records += f_receive_results(task_queue)
        
        for i, fps in enumerate(fps_records):
            print(f"[Env {i:3d}] fps: {np.mean(fps):6.2f} +/- {np.std(fps):6.2f} | median: {np.median(fps):6.2f}")

if __name__ == "__main__":
    main()
