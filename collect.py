import re
import os
import sys
import itertools as it
from time import sleep, time

import psutil
from tqdm.rich import tqdm

from iqn import IQN
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO as RPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import scenarios
from models import *
from wrapper import *
from vizdoom_utils import *
from tasks_eval import tasks
from models.training_procedure import *

# ============================== What is this ========================================
# The program to collect gameplay footage/data/performance of evaluation sessions
# ====================================================================================

# set this to -1 to execute all tasks in tasks list
task_idx = 0

# config
save_dir = "./logs"
eps_to_eval = 400
save_batch_size = 200       # (roughly) how many episodes are saved simultaneously, no promise
global_n_env_eval = 5       # number of venv (vectorized environment) to use by default
global_n_env_eval_rtss = 4  # number of venv to use for real-time semantic segmentation
env_type = SubprocVecEnv    # type of venv, SubprocVecEnv for multi-processing (recommended)
# env_type = DummyVecEnv    # don't use this unless you hate yourself a lot (or PC has no RAM)
record_pos = True

default_env_config = scenarios.FrozenDict(n_updates=1, frame_repeat=4)

# Capturing groups: 
# 1 - framestack value              (int)
# 2 - model type                    (str=["ppo", "r_ppo", "iqn"]) 
# 3 - model variant identifier      (str=["sm", "md", "lg"]) 
# 4 - model's input representation  (str=["rgb", "ss", "ss_sgb", "srgb"]) 
# 5 - model's learning rate value   (float)
extract_model_params = re.compile(
    r"^(?:[rs]([0-9]+)_)?" + \
    r"(?:((?:r_)?[ip][qp][no])_)?" + \
    r"(?:([sml][mdg])_)?" + \
    r"((?:ss_rgb)|(?:srgb)|(?:ss)|(?:rgb))_" + \
    r"([0-9\+\-\.]+[e]?[0-9\+\-]*)"
)

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

model_types = {
    "ppo"   : PPO,
    "r_ppo" : RPPO,
    "iqn"   : IQN,
}

# Tasks to perform
if task_idx >= 0:
    tasks = tasks[task_idx:task_idx+1]

env_kwargs_template = {
    "smooth_frame"  : False,
    "n_updates"     : 1,
    "frame_repeat"  : 4,
    "only_pos"      : record_pos,
    "measure_miou"  : True,
}

def main():
    global tasks

    for config, name, input_rep, seed, additonal_config, model_path in tasks:
        framestack = 0
        model_type = None

        model_info = extract_model_params.fullmatch(name)
        if model_info is not None:
            framestack_str, model_type, _, input_rep_str, _ = model_info.groups()
            if framestack_str is not None and int(framestack_str) > 1:
                framestack = int(framestack_str)
            if input_rep_str is not None:
                input_rep = scenarios.input_definitions[input_rep_str]

        path_has_decimal = model_path.find(".0")
        model_path[:path_has_decimal] if path_has_decimal > 0 else model_path
        save_name = f"{config}/{name}_{model_path}"
        ch_num = scenarios.input_definitions[input_rep]
        if framestack:
            ch_num *= framestack
        # print(ch_num, '|', framestack, '|', framestack_str, '|', input_rep, '|', input_rep_str)
        game_config = dict(config_path=scenarios.maps[config], color=True, label=True, res=(256, 144), visibility=False, add_args=bot_args)

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
        if framestack:
            n_env_eval //= framestack
        n_env_eval = max(1, n_env_eval)
        
        mem_size_est = additonal_config.get("memsize", None)
        if mem_size_est is None:
            mem_size_est = round(1300 * eps_to_eval / n_env_eval)
            if n_env_eval > 1:
                mem_size_est += max(round(mem_size_est / 10), 2600)

        env_kwargs = env_kwargs_template.copy()
        env_kwargs.update({
            "realtime_ss"   : "rtss" in save_name,
            "frame_stack"   : bool(framestack),
            "buffer_size"   : int(framestack),
            "actions"       : act_actions, 
            "input_shape"   : (ch_num, 144, 256), 
            "game_config"   : game_config,
            "seed"          : seed,
            "input_rep"     : input_rep,
            "memsize"       : mem_size_est,
        })

        env_kwargs.update(additonal_config.get("env_config", default_env_config))

        policy_kwargs = {
            "normalize_images"          : True,
            "features_extractor_class"  : CustomCNN,
            "features_extractor_kwargs" : {"features_dim" : 128}
        }

        if input_rep in (3, "srgb"):
            policy_kwargs["normalize_images"] = False

        eval_env_kwargs = env_kwargs.copy()
        eval_game_config = game_config.copy()
        eval_game_config["add_args"] = bot_args_eval
        eval_env_kwargs["game_config"] = eval_game_config
        eval_env: SubprocVecEnv | DummyVecEnv | VecEnv = make_vec_env(
            DoomBotDeathMatchCapture, n_envs=n_env_eval, seed=seed, env_kwargs=eval_env_kwargs, vec_env_cls=env_type)

        if model_path:
            if not os.path.exists(f"{save_dir}/{name}/{model_path}"):
                model_path = f"{model_path}.zip"
        else:
            model_path = "best_model.zip"
            if not os.path.exists(f"{save_dir}/{name}/{model_path}"):
                model_path = "final.zip"


        mem = psutil.virtual_memory().available / 1e9

        # Not the most elegant way to do it, but I guess it works...
        model_cls = model_types.get(model_type, RPPO)
        try:
            model = model_cls.load(f"{save_dir}/{name}/{model_path}")
        except:
            try:
                model_cls = PPO
                model = model_cls.load(f"{save_dir}/{name}/{model_path}")
            except:
                model_cls = IQN
                model = model_cls.load(f"{save_dir}/{name}/{model_path}")

        if sys.platform.lower()[:3] == "win":
            os.system("cls")
        else:
            os.system("clear")
            
        mem_after_model_alloc = psutil.virtual_memory().available / 1e9
        prob_alloc = mem - mem_after_model_alloc
        print(f"{mem_after_model_alloc:.2f} GB available, model probably occupied {prob_alloc:.2f} GB")
        print(f"Number of available buttons: {n} | Size of action space: {len(act_actions)}")
        print(f"Input shape: {env_kwargs['input_shape']}", flush=True)

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
