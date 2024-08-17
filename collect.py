import itertools as it
import os
import torch
import torch.nn as nn
import vizdoom as vzd

from time import sleep, time
from torch.optim import SGD, Adam, Adagrad

from models import *
from vizdoom_utils import *
# from models.training_procedure import *

task_idx = 0

# ============================== What is this ========================================
# The main program to run training sessions
# ====================================================================================

# Deathmatch Scenarios
maps = {
    "map1"  : os.path.join(vzd.scenarios_path, "deathmatch_simple.cfg"),
    "map1a" : os.path.join(vzd.scenarios_path, "bots_deathmatch_1_alt.cfg"),
    "map2"  : os.path.join(vzd.scenarios_path, "bots_deathmatch_2.cfg"),
    "map2s" : os.path.join(vzd.scenarios_path, "bots_deathmatch_2_shotgun.cfg"),
    "map3"  : os.path.join(vzd.scenarios_path, "bots_deathmatch_3.cfg"),
}

for k in list(maps.keys()):
    if "rtss_" not in k:
        maps["rtss_" + k] = maps[k]

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
save_validation = False
# config, lr, ep_max, save_interval, name, input_rep, n_envs, seed

input_rep_ch_num = {
    0   : 3,    # RGB
    1   : 1,    # Semantic Segmentation Mask
    2   : 4,    # RGBS (S = Semantic Segmentation Mask)
    3   : 3     # Semantic Segmentation as RGB
}



tasks = [
    ("map1"         , 3e-4,      1, 1, "ss_rgb_3e-4",    2, 4, 2050808),
    ("map1"         , 3e-4,      1, 1, "rgb_3e-4",       0, 4, 2050808),
    ("map1"         , 3e-4,      1, 1, "ss_3e-4",        1, 4, 2050808),
    ("map1"         , 1e-4,      1, 1, "ss_rgb_1e-4",    2, 4, 2050808),
    ("map1"         , 1e-4,      1, 1, "rgb_1e-4",       0, 4, 2050808),
    ("map1"         , 1e-4,      1, 1, "ss_1e-4",        1, 4, 2050808),
    ("map1"         , 9e-4,      1, 1, "ss_rgb_9e-4",    2, 4, 2050808),
    ("map1"         , 9e-4,      1, 1, "rgb_9e-4",       0, 4, 2050808),
    ("map1"         , 9e-4,      1, 1, "ss_9e-4",        1, 4, 2050808),
    ("map1"         , 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808),
    ("map1"         , 1e-5,      1, 1, "ss_1e-5",        1, 4, 2050808),
    ("map2s"        , 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808),
    ("map2s"        , 1e-5,      1, 1, "ss_1e-5",        1, 4, 2050808),
    ("map3"         , 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808),
    ("map3"         , 1e-5,      1, 1, "ss_1e-5",        1, 4, 2050808),
    ("rtss_map1"    , 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808),
    ("rtss_map1"    , 1e-5,      1, 1, "ss_1e-5",        1, 4, 2050808),
    ("rtss_map2s"   , 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808),
    ("rtss_map2s"   , 1e-5,      1, 1, "ss_1e-5",        1, 4, 2050808),
    ("rtss_map3"    , 1e-3,      1, 1, "ss_1e-3",        1, 4, 2050808),
    ("rtss_map3"    , 1e-5,      1, 1, "ss_1e-5",        1, 4, 2050808),
    ("map1"         , 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808),
    ("rtss_map1"    , 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808),
    ("map2s"        , 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808),
    ("rtss_map2s"   , 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808),
    ("map3"         , 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808),
    ("rtss_map3"    , 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808),
][task_idx:task_idx+1]

tasks = [("rtss_map2s"   , 1e-3,      1, 1, "ss_rgb_1e-3",    2, 4, 2050808)]

# tasks = [
#     ("map1", 3e-4,      1, 1, "ssr_3e-4",       3, 4, 2050808),
#     ("map1", 3e-4,      1, 1, "ss_rgb_3e-4",    2, 4, 2050808),
#     ("map1", 3e-4,      1, 1, "rgb_3e-4",       0, 4, 2050808),
#     ("map1", 3e-4,      1, 1, "ss_3e-4",        1, 4, 2050808),
#     ("map1", 1e-4,      1, 1, "ssr_1e-4",       3, 4, 2050808),
#     ("map1", 1e-4,      1, 1, "ss_rgb_1e-4",    2, 4, 2050808),
#     ("map1", 1e-4,      1, 1, "rgb_1e-4",       0, 4, 2050808),
#     ("map1", 1e-4,      1, 1, "ss_1e-4",        1, 4, 2050808),
# ][task_idx:task_idx+1]

# tasks = [
#     ("map1", 3e-3,      1, 1, "ss_rgb_3e-3",    2, 4, 2050808),
#     ("map1", 3e-3,      1, 1, "rgb_3e-3",       0, 4, 2050808),
#     ("map1", 3e-3,      1, 1, "ss_3e-3",        1, 4, 2050808),
#     ("map1", 3e-4,      1, 1, "ss_rgb_3e-4",    2, 4, 2050808),
#     ("map1", 3e-4,      1, 1, "rgb_3e-4",       0, 4, 2050808),
#     ("map1", 3e-4,      1, 1, "ss_3e-4",        1, 4, 2050808),
# ][5:6]


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
    global save_validation, tasks, agent
    # DEVICE = check_gpu()
    # ep_max (maximum training episode) would overwrite the epoch_num (number of epochs) setting
    for config, lr, ep_max, save_interval, name, input_rep, n_envs, seed in tasks:
        save_name = f"{config}_small_{name}"
        ch_num = input_rep_ch_num[input_rep]
        game_config = dict(config_path=maps[config], color=True, label=True, res=(256, 144), visibility=False, add_args=bot_args)

        game = create_game(**game_config)
        n = game.get_available_buttons_size()
        act_actions = (tuple(x) for x in it.product([False, True], repeat=n) if any(x))
        act_actions = (x for x in act_actions if not (x[2] and x[3]))   # filter out strafe left (3) and right (4) simultaneously
        act_actions = (x for x in act_actions if not (x[4] and x[5]))   # filter out turn   left (4) and right (5) simultaneously
        act_actions = [x for x in act_actions if not (x[0]         )]   # filter out actions with attack (0)
        act_actions = [tuple([True] + [False] * (n-1))] + act_actions   # add back one action that performs attack (0) only

        if False:
            mem_size = 100_000 if (ep_max <= 100) else 300_000
            
            agent = CombatAgent(
                device=DEVICE, mem_size=mem_size, action_num=len(act_actions), nav_action_num=len(nav_actions), 
                discount=0.99, lr=lr, dropout=0.5, loss=nn.HuberLoss, act_wd=0, nav_wd=0, optimizer=Adam, 
                state_len=10, act_model=DRQNv2, nav_model=DQNv1, eps=1., eps_decay=0.99995, ch_num=ch_num, name=name)
            
            if save_validation:
                agent.name = 'test'
                file_name = agent.save_models(-1)
                file_size = os.path.getsize(file_name)
                os.remove(file_name)
                file_size /= (1024*1024*1024)
                agent.name = name
                save_validation = False
                print(f"Save validation passed, each save occupies {file_size:.2f} GiB of storage space")
            
            print(f"Memory capacity: {mem_size} | Action space: {len(act_actions)} (combat), {len(nav_actions)} (nav)")

        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.callbacks import EvalCallback
        from tqdm import tqdm

        from wrapper import DoomBotDeathMatch, DoomBotDeathMatchCapture

        game.close()
        del game

        eps_to_eval = 20
        n_env_eval = 4
        p = tqdm(total=eps_to_eval)
        eval_callback = lambda locals_dict, _: (scores.append(locals_dict['info']['r']), p.update()) if locals_dict['done'] else None

        mem_size_est = round(1300 * eps_to_eval / n_env_eval)
        if n_env_eval > 1:
            mem_size_est += max(round(mem_size_est / 10), 2600)
        
        # mem_size_est = -1

        env_kwargs = {
            # "set_map"       : "m",
            "smooth_frame"  : True,
            "realtime_ss"   : "rtss" in save_name,
            "actions"       : act_actions, 
            "input_shape"   : (ch_num, 144, 256), 
            "game_config"   : game_config,
            "frame_repeat"  : 4,
            "seed"          : seed,
            "input_rep"     : input_rep,
            "memsize"       : mem_size_est
        }

        policy_kwargs = {
            "normalize_images"          : True,
            "features_extractor_class"  : CustomCNN,
            "features_extractor_kwargs" : {"features_dim" : 128}
        }

        if input_rep == 3:
            policy_kwargs["normalize_images"] = False

        t = time()
        env_type = SubprocVecEnv
        eval_env_kwargs = env_kwargs.copy()
        eval_game_config = game_config.copy()
        eval_game_config["add_args"] = bot_args_eval
        eval_env_kwargs["game_config"] = eval_game_config
        eval_env = make_vec_env(DoomBotDeathMatchCapture, n_envs=n_env_eval, seed=seed, env_kwargs=eval_env_kwargs, vec_env_cls=env_type)

        from stable_baselines3 import PPO
        model = PPO.load(f"./logs/{name}/best_model.zip")
        
        print(f"Number of available buttons: {n} | Size of action space: {len(act_actions)}", flush=True)
        p.update(0)
        scores = []
        evaluate_policy(model, eval_env, eps_to_eval, callback=eval_callback)
        p.set_description("Finished evaluations")
        p.close()

        if not os.path.exists(f"./logs/{save_name}"):
            os.makedirs(f"./logs/{save_name}")

        scores = np.array(scores, dtype=np.int64)
        np.save(f"./logs/{save_name}/performance.npy", scores)
        print(f"Results saved to ./logs/{save_name}/performance.npy", flush=True)
        print(f"Scores: {scores.tolist()}\n{scores.mean():.2f} +/- {scores.std():.3f} | min: {np.min(scores):2d} | Q1: {int(np.percentile(scores, 25)):2d} | Q2: {int(np.percentile(scores, 50)):2d} | Q3: {int(np.percentile(scores, 75)):2d} | max: {np.max(scores):2d}")

        target_remotes = eval_env._get_target_remotes(None)
        fps_records = []
        if eps_to_eval <= 50:
            for i, remote in enumerate(target_remotes):
                remote.send(("env_method", ("save", [f"./logs/{save_name}/record_{i}.npz"], {})))
            fps_records += [remote.recv() for remote in target_remotes]
        else:
            queue = []
            queue_limit = (250 // (eps_to_eval // eps_to_eval))
            for i, remote in enumerate(target_remotes):
                remote.send(("env_method", ("save", [f"./logs/{save_name}/record_{i}.npz"], {})))
                queue.append(remote)
                if len(queue) >= queue_limit:
                    fps_records += [remote.recv() for remote in queue]
                    queue = []
            
            if len(queue):
                fps_records += [remote.recv() for remote in queue]
        
        for i, fps in enumerate(fps_records):
            print(f"[Env {i:3d}] fps: {np.mean(fps):6.2f} +/- {np.std(fps):6.2f} | median: {np.median(fps):6.2f}")
        

if __name__ == "__main__":
    main()
