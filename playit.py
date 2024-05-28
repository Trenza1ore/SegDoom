import vizdoom as vzd
import os
import numpy as np
from matplotlib.image import imsave
from matplotlib.pyplot import imshow, savefig, title, tight_layout
from vizdoom_utils import resize_cv_linear, create_game

# ============================== What is this ========================================
# A small program that allows human players to play (and record) a scenario themselves
# ====================================================================================

# config
is_recording = True
save_state = True
#res = (256, 144)
res = (640, 400)
resize_res = (128, 72)

config_file_path = os.path.join(vzd.scenarios_path, "deathmatch_hugo.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "hanger_test.cfg")
#config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor_hugo.cfg")

# create folder for holding screenshots
if is_recording:
    if not os.path.isdir("screenshots"):
        os.mkdir("screenshots")
game = create_game(config_file_path, color=True, depth=True, label=True, res=res, visibility=True, asyn=True, spec=True)
states = []

# play the game as a human (while taking screenshots)
while True:
    game.new_episode()
    i = 0
    if is_recording:
        tight_layout()
    while not game.is_episode_finished():
        if is_recording:
            i += 1
            state = game.get_state()
            frame = state.screen_buffer
            frame_depth = state.depth_buffer
            frame_label = state.labels_buffer
            if save_state:
                states.append(state)
            #frame_lr = resize_cv_linear(frame, resize_res)
            imsave(f"screenshots/{i}.png", frame)
            #imsave(f"screenshots/low_res_{i}.png", frame_lr)
            imsave(f"screenshots/depth_{i}.png", frame_depth)
            imshow(frame_label, cmap="gist_ncar")
            title(np.unique(frame_label))
            savefig(f"screenshots/label_{i}.png")
            #imsave(f"screenshots/label_{i}.png", frame_label, cmap="gist_ncar")
        game.advance_action()
    print(f"You scored: {game.get_total_reward():.2f}")