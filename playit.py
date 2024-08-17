import vizdoom as vzd
import os
import time
import numpy as np
from matplotlib.image import imsave
from matplotlib.pyplot import imshow, savefig, title, tight_layout
from vizdoom_utils import resize_cv_linear, create_game, RewardTracker
from vizdoom.vizdoom import GameVariable

# ============================== What is this ========================================
# A small program that allows human players to play (and record) a scenario themselves
# ====================================================================================

# config
print_score_board = False
is_recording = True
save_state = "recording.pkl"
#res = (256, 144)
res = (640, 400)
res = (1920, 1080)
resize_res = (128, 72)

config_file_path = os.path.join(vzd.scenarios_path, "deathmatch_simple.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "bots_deathmatch_2_shotgun.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "bots_deathmatch_3.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "hanger_test.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")

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

game = create_game(config_file_path, color=True, depth=True, label=True, res=res, visibility=True, asyn=False, spec=True, add_args=add_args)
game.set_doom_map("map01")
# game.send_game_command("iddqd")
game.new_episode()
states = []

time.sleep(7)
try:
    # play the game as a human (while taking screenshots)
    while True:
        game.new_episode()
        game.send_game_command("removebots")
        for _ in range(8):
            game.send_game_command('addbot')
        #game.send_game_command("idspispopd")
        #game.send_game_command("idbeholdi")
        # game.send_game_command("bot_observer 1")
        print(game.get_game_args())
        r = RewardTracker(game)
        i = 0
        if is_recording:
            tight_layout()
        while not game.is_episode_finished():
            if is_recording:
                i += 1
                state = game.get_state()
                # frame = state.screen_buffer
                # frame_depth = state.depth_buffer
                # frame_label = state.labels_buffer
                # print(state.game_variables)
                if save_state:
                    states.append(state)
                # #frame_lr = resize_cv_linear(frame, resize_res)
                # imsave(f"screenshots/{i}.png", frame)
                # #imsave(f"screenshots/low_res_{i}.png", frame_lr)
                # imsave(f"screenshots/depth_{i}.png", frame_depth)
                # imshow(frame_label, cmap="gist_ncar")
                # title(np.unique(frame_label))
                # savefig(f"screenshots/label_{i}.png")
                # #imsave(f"screenshots/label_{i}.png", frame_label, cmap="gist_ncar")
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
            # print(f"\rWeapon: {int(game.get_game_variable(GameVariable.FRAGCOUNT))}", end='')
            game.advance_action()
            r.update()
        print(f"You scored: {int(game.get_game_variable(GameVariable.FRAGCOUNT))}")
except Exception as e:
    raise e
finally:
    import pickle
    if save_state:
        with open(save_state, "wb") as f:
            pickle.dump(states, f, -1)
        print(f"States saved to {save_state}")
