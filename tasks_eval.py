import sys

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

save_dir = "./logs"
eps_to_eval = 400
save_batch_size = 200       # (roughly) how many episodes are saved simultaneously, no promise
global_n_env_eval = 20      # number of venv (vectorized environment) to use by default
global_n_env_eval_rtss = 4  # number of venv to use for real-time semantic segmentation
env_type = SubprocVecEnv    # type of venv, SubprocVecEnv for multi-processing (recommended)
# env_type = DummyVecEnv    # don't use this unless you hate yourself a lot (or PC has no RAM)

# My Ubuntu machine has less RAM & VRAM
if sys.platform.startswith("linux"):
    global_n_env_eval = 5
    global_n_env_eval_rtss = 4

env_kwargs_template = {
    "smooth_frame"  : False,
    "n_updates"     : 1,
    "frame_repeat"  : 4,
    "only_pos"      : True,
    "measure_miou"  : True,
}

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

    # 40-42
    ("map1"         , "ss_1e-3",            1, 2050808, {}, ''),       # 40
    ("map2s"        , "ss_1e-3",            1, 2050808, {}, ''),       # 41
    ("map3"         , "ss_1e-3",            1, 2050808, {}, ''),       # 42

    # 43-45
    ("map1"         , "ss_rgb_1e-3",        1, 2050808, {}, ''),       # 43
    ("map2s"        , "ss_rgb_1e-3",        1, 2050808, {}, ''),       # 44
    ("map3"         , "ss_rgb_1e-3",        1, 2050808, {}, ''),       # 45

    # 46-51
    ("map1"         , "ppo_ss_1e-3",        1, 2050808, {}, 'final'),
    ("map2s"        , "ppo_ss_1e-3",        1, 2050808, {}, 'final'),
    ("map3"         , "ppo_ss_1e-3",        1, 2050808, {}, 'final'),
    ("map1"         , "ppo_ss_1e-3",        1, 2050808, {}, 'best_model_20'),
    ("map2s"        , "ppo_ss_1e-3",        1, 2050808, {}, 'best_model_20'),
    ("map3"         , "ppo_ss_1e-3",        1, 2050808, {}, 'best_model_20'),

    # 52-57
    ("map1"         , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'final'),
    ("map2s"        , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'final'),
    ("map3"         , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'final'),
    ("map1"         , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best_model_26'),
    ("map2s"        , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best_model_26'),
    ("map3"         , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best_model_26'),

    # 58-63
    ("map1"         , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'final'),
    ("map2s"        , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'final'),
    ("map3"         , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'final'),
    ("map1"         , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'best_model_25.0_121'),
    ("map2s"        , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'best_model_25.0_121'),
    ("map3"         , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'best_model_25.0_121'),

    # 64-72
    ("map1"         , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'final'),
    ("map2s"        , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'final'),
    ("map3"         , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'final'),
    ("map1"         , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'best_model_20.0_108'),
    ("map2s"        , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'best_model_20.0_108'),
    ("map3"         , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'best_model_20.0_108'),
    ("map1"         , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'best_model_21.0_106'),
    ("map2s"        , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'best_model_21.0_106'),
    ("map3"         , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'best_model_21.0_106'),

    # 73-87
    ("map1"         , "rgb_1e-3",           1, 2050808, {}, ''),
    ("map2s"        , "rgb_1e-3",           1, 2050808, {}, ''),
    ("map3"         , "rgb_1e-3",           1, 2050808, {}, ''),
    ("map1"         , "rgb_1e-4",           1, 2050808, {}, ''),
    ("map2s"        , "rgb_1e-4",           1, 2050808, {}, ''),
    ("map3"         , "rgb_1e-4",           1, 2050808, {}, ''),
    ("map1"         , "rgb_1e-5",           1, 2050808, {}, ''),
    ("map2s"        , "rgb_1e-5",           1, 2050808, {}, ''),
    ("map3"         , "rgb_1e-5",           1, 2050808, {}, ''),
    ("map1"         , "rgb_3e-4",           1, 2050808, {}, ''),
    ("map2s"        , "rgb_3e-4",           1, 2050808, {}, ''),
    ("map3"         , "rgb_3e-4",           1, 2050808, {}, ''),
    ("map1"         , "rgb_9e-4",           1, 2050808, {}, ''),
    ("map2s"        , "rgb_9e-4",           1, 2050808, {}, ''),
    ("map3"         , "rgb_9e-4",           1, 2050808, {}, ''),

    # 88-90 (I'm an idiot, this is meaningless)
    ("rtss_map1"    , "rgb_9e-4",           1, 2050808, {}, ''),
    ("rtss_map2s"   , "rgb_9e-4",           1, 2050808, {}, ''),
    ("rtss_map3"    , "rgb_9e-4",           1, 2050808, {}, ''),

    # 91-93
    ("rtss_map1"    , "ppo_ss_1e-3",        1, 2050808, {}, 'final'),
    ("rtss_map2s"   , "ppo_ss_1e-3",        1, 2050808, {}, 'final'),
    ("rtss_map3"    , "ppo_ss_1e-3",        1, 2050808, {}, 'final'),

    # 94-96
    ("rtss_map1"    , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'final'),
    ("rtss_map2s"   , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'final'),
    ("rtss_map3"    , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'final'),

    # 97-99
    ("rtss_map1"    , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best'),
    ("rtss_map2s"   , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best'),
    ("rtss_map3"    , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best'),

    # 100-102
    ("rtss_map1"    , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'final'),
    ("rtss_map2s"   , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'final'),
    ("rtss_map3"    , "s4_ppo_ss_1e-3",     1, 2050808, {}, 'final'),

    # 103-105
    ("rtss_map1"    , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'final'),
    ("rtss_map2s"   , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'final'),
    ("rtss_map3"    , "s4_ppo_ss_5e-4",     1, 2050808, {}, 'final'),

    # 106-108
    ("map1"         , "s4_ppo_ss_rgb_5e-4", 1, 2050808, {}, 'final'),
    ("map2s"        , "s4_ppo_ss_rgb_5e-4", 1, 2050808, {}, 'final'),
    ("map3"         , "s4_ppo_ss_rgb_5e-4", 1, 2050808, {}, 'final'),

    # 109-111
    ("map1"         , "s4_ppo_ss_rgb_5e-4", 1, 2050808, {}, 'best_model_23.0_115'),
    ("map2s"        , "s4_ppo_ss_rgb_5e-4", 1, 2050808, {}, 'best_model_23.0_115'),
    ("map3"         , "s4_ppo_ss_rgb_5e-4", 1, 2050808, {}, 'best_model_23.0_115'),

    # 112-114
    ("map1"         , "s4_ppo_ss_rgb_1e-4", 1, 2050808, {}, 'final'),
    ("map2s"        , "s4_ppo_ss_rgb_1e-4", 1, 2050808, {}, 'final'),
    ("map3"         , "s4_ppo_ss_rgb_1e-4", 1, 2050808, {}, 'final'),

    # 115-117
    ("map1"         , "s4_ppo_ss_rgb_1e-4", 1, 2050808, {}, 'best_model_23.0_115'),
    ("map2s"        , "s4_ppo_ss_rgb_1e-4", 1, 2050808, {}, 'best_model_23.0_115'),
    ("map3"         , "s4_ppo_ss_rgb_1e-4", 1, 2050808, {}, 'best_model_23.0_115'),

    # 118-120
    ("map1"         , "r1_r_ppo_sl_ss_5e-4",1, 2050808, {}, 'final'),
    ("map2s"        , "r1_r_ppo_sl_ss_5e-4",1, 2050808, {}, 'final'),
    ("map3"         , "r1_r_ppo_sl_ss_5e-4",1, 2050808, {}, 'final'),

    # 121-123
    ("map1"         ,"r1_r_ppo_Dsl_ss_5e-4",1, 2050808, {}, 'final'),
    ("map2s"        ,"r1_r_ppo_Dsl_ss_5e-4",1, 2050808, {}, 'final'),
    ("map3"         ,"r1_r_ppo_Dsl_ss_5e-4",1, 2050808, {}, 'final'),

    # 124-126
    ("rtss_map1"    , "r1_r_ppo_sl_ss_5e-4",1, 2050808, {}, 'final'),
    ("rtss_map2s"   , "r1_r_ppo_sl_ss_5e-4",1, 2050808, {}, 'final'),
    ("rtss_map3"    , "r1_r_ppo_sl_ss_5e-4",1, 2050808, {}, 'final'),

    # 127-129
    ("rtss_map1"    ,"r1_r_ppo_Dsl_ss_5e-4",1, 2050808, {}, 'final'),
    ("rtss_map2s"   ,"r1_r_ppo_Dsl_ss_5e-4",1, 2050808, {}, 'final'),
    ("rtss_map3"    ,"r1_r_ppo_Dsl_ss_5e-4",1, 2050808, {}, 'final'),

    # 130-133
    ("map1a"        , "rgb_9e-4",           1, 2050808, {}, ''),
    ("map1a"        , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best'),
    ("map1w"        , "rgb_9e-4",           1, 2050808, {}, ''),
    ("map1w"        , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best'),

    # 134-137
    ("rtss_map1a"   , "rgb_9e-4",           1, 2050808, {}, ''),
    ("rtss_map1a"   , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best'),
    ("rtss_map1w"   , "rgb_9e-4",           1, 2050808, {}, ''),
    ("rtss_map1w"   , "ppo_ss_rgb_1e-3",    1, 2050808, {}, 'best'),
]