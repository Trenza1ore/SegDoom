import os
import time
import concurrent.futures

import numpy as np
import vizdoom as vzd
from tqdm.rich import trange

from scenarios import maps
from vizdoom_utils import create_game

# ============================== What is this ========================================
# A small program that allows human players to play (and record) a scenario themselves
# ====================================================================================

# config
res = (160, 120)
n_episodes = 4_000
map_to_play = "map1"
game_seed = 2050808

add_args = " ".join([
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
    "+sv_cheats 1",
    "+sv_noexit 1"
])
# "map1", "map2", "map2s", 
for map_to_play in ["map3", "map2s", "map1"]:
    game = create_game(maps[map_to_play], color=False, depth=False, label=True, res=res, visibility=False, add_args=add_args)
    game.set_seed(game_seed)
    print(f"Evaluating {map_to_play} ({maps[map_to_play]})", flush=True)

    data = np.zeros(n_episodes, dtype=np.int64)
    t = trange(n_episodes, miniters=1)

    for i in t:
        game.send_game_command("removebots")
        for _ in range(9):
            game.send_game_command('addbot')
        game.send_game_command("iddqd")
        game.send_game_command("bot_observer 1")
        while not game.is_episode_finished():
            game.advance_action(tics=6000)
        sv_state = game.get_server_state()
        data[i] = max(sv_state.players_frags)
        game.new_episode()
        if i % 10 is 0:
            d = data[:i+1]
            t.set_description(f"{np.mean(d):5.2f}")# | min: {np.min(d):2d} | Q1: {int(np.percentile(d, 25)):2d} | Q2: {int(np.percentile(d, 50)):2d} | Q3: {int(np.percentile(d, 75)):2d} | max: {np.max(d):2d} |")

    data = np.asarray(data, dtype=np.int64)
    print(f"\n{data.mean():.3f} +/- {data.std():.3f}\nmin: {int(data.min()):2d} | max: {int(data.max()):2d}\nQ1: {int(np.percentile(data, 25)):2d} | Q2: {int(np.percentile(data, 50)):2d} | Q3: {int(np.percentile(data, 75)):2d}")
    if not os.path.exists(f"logs/{map_to_play}/bot"):
        os.makedirs(f"logs/{map_to_play}/bot")
    np.save(f"logs/{map_to_play}/bot/performance.npy", data)