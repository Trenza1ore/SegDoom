import os
import time

import numpy as np
from matplotlib.image import imsave
from vizdoom.vizdoom import GameVariable
from matplotlib.pyplot import imshow, savefig, title, tight_layout

from scenarios import maps
from vizdoom_utils import resize_cv_linear, create_game, RewardTracker

# ============================== What is this ========================================
# A small program that allows human players to play (and record) a scenario themselves
# ====================================================================================

# config
print_score_board = False
is_recording = True
save_state = "recording.pkl"
#res = (256, 144)
res = (1920, 1080)
resize_res = (128, 72)
n_bots = 0
n_games = 1                # -1 for infinite
delay_after_run = 3
async_game = False
map_to_play = "map1w"
seed = 20508
# precision_action_buffer = [
#     [False] * 5 + [True],
#     [False] * 5 + [True],
#     [False] * 5 + [True],
# ]

# On Windows (not a main focus of ViZDoom), a 3-10 seconds delay is sometimes required
# to run ViZDoom properly, otherwise ViZDoom game instance is never initialized properly,
# game window would freeze indefinitely. This bug only affects software renderer, game 
# logic works fine... Also the delay must be not too short AND NOT TOO LONG (why?).
# Notes: changing ANY line in this file may lead to the wait value becoming invalid
wait_for_vizdoom_delay = 6

config_file_path = maps[map_to_play]

# create folder for holding screenshots
if is_recording:
    if not os.path.isdir("screenshots"):
        os.mkdir("screenshots")

add_args = " ".join([
    "-host 1",
    "-deathmatch",
    "+viz_nocheat 0",
    "+cl_run 1",
    # "+name AGENT",
    # "+colorset 0",
    "+sv_forcerespawn 1",
    "+sv_respawnprotect 1",
    # "+sv_losefrag 0",
    "+sv_nocrouch 1",
    "+sv_cheats 1",
    "+sv_noexit 1"
])

game = create_game(config_file_path, color=True, depth=True, label=True, res=res, visibility=True, asyn=async_game, spec=True, add_args=add_args)
# game.set_doom_map("map01")
game.set_seed(seed)
# game.send_game_command("iddqd")
game.new_episode()
states = []

time.sleep(wait_for_vizdoom_delay)
try:
    # play the game as a human (while taking screenshots)
    games_to_play = [1] * n_games if n_games >= 0 else True
    while games_to_play:
        game.new_episode()
        game.send_game_command("removebots")
        for _ in [0] * n_bots:
            game.send_game_command('addbot')
        #game.send_game_command("idspispopd")
        #game.send_game_command("idbeholdi")
        # game.send_game_command("bot_observer 1")
        print(game.get_game_args())
        r = RewardTracker(game)
        i = 0
        if is_recording:
            pass
            # tight_layout()
        while not game.is_episode_finished():
            if is_recording:
                i += 1
                state = game.get_state()
                frame = state.screen_buffer
                # frame_depth = state.depth_buffer
                # frame_label = state.labels_buffer
                # print(state.game_variables)
                if save_state:
                    states.append(state)
                # #frame_lr = resize_cv_linear(frame, resize_res)
                imsave(f"screenshots/{map_to_play}_{i}.png", frame)
                # #imsave(f"screenshots/low_res_{i}.png", frame_lr)
                # imsave(f"screenshots/depth_{i}.png", frame_depth)
                # imshow(frame_label, cmap="jet")
                # title(np.unique(frame_label))
                # savefig(f"screenshots/label_{i}.png")
                # #imsave(f"screenshots/label_{i}.png", frame_label, cmap="jet")
            if print_score_board:
                sv_state = game.get_server_state()
                print(
                    [
                    i for i in zip(
                    sv_state.players_names,
                    sv_state.players_frags, 
                    sv_state.players_in_game)
                    if len(i[0])
                    ], end="\r", flush=True
                )
            #print(f"\rDmg: {game.get_game_variable(GameVariable.DAMAGECOUNT):.2f}", end="", flush=True)
            # print(f"\rWeapon: {int(game.get_game_variable(GameVariable.SELECTED_WEAPON))}", end='')
            game.advance_action()
            r.update()
        print(f"You scored: {int(game.get_game_variable(GameVariable.FRAGCOUNT))}")
        games_to_play.pop()
        if games_to_play:
            time.sleep(delay_after_run)
except Exception as e:
    raise e
finally:
    import pickle
    if save_state:
        with open(save_state, "wb") as f:
            pickle.dump(states, f, 5)
        print(f"States saved to {save_state}")
