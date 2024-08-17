from random import randrange
from time import sleep, time

import vizdoom as vzd
from tqdm import trange
from gymnasium import Env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from models import *
from stats import *
from vizdoom_utils import *

# ============================== What is this ========================================
# Helper functions for running training sessions for an RL agent
# ====================================================================================

class inf_ep_max:
    """a class created to represent max value as Python 3's int has no such thing
    """
    def __eq__(self, __value: object) -> bool:
        return False
    def __gt__(self, __value: object) -> bool:
        return True
    def __ge__(self, __value: object) -> bool:
        return True
    def __lt__(self, __value: object) -> bool:
        return False
    def __le__(self, __value: object) -> bool:
        return False

def train_agent(game: vzd.vizdoom.DoomGame, 
                agent: CombatAgent, action_space: list, nav_action_space: list, 
                skip_training: bool=False, discord: bool=False, epoch_num: int=3, 
                frame_repeat: int=4, epoch_step: int=500, load_epoch: int=-1, 
                downsampler=(lambda x, y: x), res: tuple[int, int]=(128, 72), 
                random_runs: bool=False, ep_max: int=0, save_interval: int=0,
                label_def: dict=None
                ) -> tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]:
    """Runs a training session of an RL agent, with discord webhook support for sending statistics of training scores after each epoch of training.
    Written in an unstructured manner to have marginal (yet existing) performance gains. Deadly corridor is better trained with train_agent_corridor.
    
    Args:
        game (vzd.vizdoom.DoomGame): vizdoom game instance.
        agent (CombatAgent): the RL agent to train.
        action_space (list): action space for the combat model.
        nav_action_space (list): action space for the navigation model.
        skip_training (bool, optional): whether to skip training (for debug only). Defaults to False.
        discord (bool, optional): whether to have discord webhook send training stats and plots after every epoch. Defaults to False.
        epoch_num (int, optional): number of epoches to train (not necessarily reached if ep_max is set). Defaults to 3.
        frame_repeat (int, optional): the number of frames to repeat. Defaults to 4.
        epoch_step (int, optional): minimum number of steps in an epoch (the last episode in an epoch is always finished). Defaults to 500.
        load_epoch (int, optional): legacy option, use the load_models method of agent instance instead. Defaults to -1.
        downsampler (_type_, optional): downsampling algorithm to use, bilinear intepolation is the most balanced but nearest is slightly faster. Defaults to resize_cv_linear.
        res (tuple[int, int], optional): downsampling algorithm's target resolution. Defaults to (128, 72).
        random_runs (bool, optional): whether to include short, purely randomized episodes between epoches, minimum steps is set to 500. Defaults to False.
        ep_max (int, optional): maximum episodes for this training session, checked after every epoch and overwrites epoch_num, 0=disable. Defaults to 0.
        save_interval (int, optional): automatically save the models and replay memory every save_interval epoches, 0=disable. Defaults to 0.

    Returns:
        tuple[CombatAgent, vzd.vizdoom.DoomGame, list[float]]: returns the agent, game instance and training scores
    """    
    
    if save_interval == 0:
        save_interval = epoch_num+1
    
    if ep_max <= 0:
        ep_max = inf_ep_max()
    
    extract_frame = (lambda state: downsampler(state.screen_buffer, res)) \
        if agent.memory.ch_num < 4 else \
            (lambda state: np.dstack([state.screen_buffer, semseg(state, label_def)]))
        
    all_scores = [[], []]
    train_quartiles = [[], [], [], []]
    if (not skip_training):
        bot = discord_bot(extra=f"deathmatch '{agent.name}' lr={agent.lr:.8f}")
        try:
            start = time()
            if load_epoch < 0:
                epoch_start = 0
            else:
                epoch_start = load_epoch
            
            action_space_size = len(action_space)
            nav_action_space_size = len(nav_action_space)
            
            print("Initial filling of replay memory with random actions")
            terminated = False
            game.new_episode()
            reward_tracker = RewardTracker(game)
            
            for _ in trange(500, miniters=10):
                state = game.get_state()
                game_variables = state.game_variables
                # frame = downsampler(state.screen_buffer, res)
                frame = extract_frame(state)

                # counter-intuitively, this is actually faster than creating a list of names
                is_combat = False
                for label in state.labels[:-1]:
                    if label.object_name == "DoomPlayer":
                        is_combat = True
                        break
                
                if is_combat:
                    action = randrange(0, action_space_size)
                    reward = game.make_action(nav_action_space[action], frame_repeat) + reward_tracker.update()
                else:
                    action = randrange(0, nav_action_space_size)
                    reward = game.make_action(nav_action_space[action], frame_repeat) + reward_tracker.update()
                
                agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                
                terminated = game.is_episode_finished()
                
                if terminated:
                    game.new_episode()
                    reward_tracker.reset_last_vars()
            
            for epoch in range(epoch_start, epoch_num):
                
                if random_runs:
                    print("Filling of replay memory with random actions")
                    terminated = False
                    
                    for _ in trange(500, miniters=10):
                        state = game.get_state()
                        game_variables = state.game_variables
                        # frame = downsampler(state.screen_buffer, res)
                        frame = extract_frame(state)
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        is_combat = False
                        for label in state.labels[:-1]:
                            if label.object_name == "DoomPlayer":
                                is_combat = True
                                break
                        
                        if is_combat:
                            action = randrange(0, action_space_size)
                            reward += game.make_action(action_space[action], frame_repeat)
                        else: 
                            action = randrange(0, nav_action_space_size)
                            reward += game.make_action(nav_action_space[action], frame_repeat)
                        
                        agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                        
                        terminated = game.is_episode_finished()
                        
                        if terminated:
                            game.new_episode()
                            reward_tracker.reset_last_vars()
                    
                    while not terminated:
                        state = game.get_state()
                        frame = extract_frame(state)
                        #frame = np.dstack([state.screen_buffer, semseg(state, label_def)])
                        game_variables = state.game_variables
                        
                        # counter-intuitively, this is actually faster than creating a list of names
                        is_combat = False
                        for label in state.labels[:-1]:
                            if label.object_name == "DoomPlayer":
                                is_combat = True
                                break
                        
                        if is_combat:
                            action = randrange(0, action_space_size)
                            reward += game.make_action(action_space[action], frame_repeat)
                        else: 
                            action = randrange(0, nav_action_space_size)
                            reward += game.make_action(nav_action_space[action], frame_repeat)
                        
                        agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                        
                        terminated = game.is_episode_finished()
                
                game.new_episode()
                reward_tracker.reset_last_vars()
                train_scores = []
                train_kills = []
                print(f"\n==========Epoch {epoch+1:3d}==========")
                
                terminated = False
                
                for _ in trange(epoch_step, miniters=10):
                    state = game.get_state()
                    frame = extract_frame(state)
                    # frame = np.dstack([state.screen_buffer, semseg(state, label_def)])
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels[:-1]:
                        if label.object_name == "DoomPlayer":
                            is_combat = True
                            break
                    
                    action = agent.decide_move(frame, is_combat)
                    reward += game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat) + reward_tracker.update()
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                    agent.train()
                    
                    terminated = game.is_episode_finished()
                    
                    if terminated:
                        train_scores.append(game.get_total_reward() + reward_tracker.total_reward)
                        train_kills.append(game_variables[1])
                        game.new_episode()
                        reward_tracker.reset_last_vars()
                
                tmp = 0
                while not terminated:
                    state = game.get_state()
                    frame = extract_frame(state)
                    # frame = np.dstack([state.screen_buffer, semseg(state, label_def)])
                    game_variables = state.game_variables
                    
                    # counter-intuitively, this is actually faster than creating a list of names
                    is_combat = False
                    for label in state.labels[:-1]:
                        if label.object_name == "DoomPlayer":
                            is_combat = True
                            break
                    
                    action = agent.decide_move(frame, is_combat)
                    reward = game.make_action(action_space[action] if is_combat else nav_action_space[action], frame_repeat) + reward_tracker.update()
                    
                    agent.add_mem(frame, action, reward, (is_combat, terminated, *game_variables[-3:]))
                    agent.train()
                    tmp += 1
                    print(tmp, end="\r")
                    
                    if (terminated := game.is_episode_finished()):
                        train_scores.append(game.get_total_reward() + reward_tracker.total_reward)
                        train_kills.append(game_variables[1])
                        # train_frags.append()
                        reward_tracker.reset_last_vars()
                
                # Save statistics
                all_scores[0].extend(train_scores)
                all_scores[1].extend(train_kills)
                
                stats = plot_stat(train_scores, all_scores, train_quartiles, epoch, agent, bot, epoch_start)
                
                duration = int(time()-start);
                timer = f"{duration//60:d} min {duration%60:d} sec"
                print(timer)
                
                if discord:
                    bot.send_string(stats+'\n'+timer)
                    bot.send_img(epoch)
                
                # Save models after epoch
                agent.save_models(epoch) if not (epoch+1)%save_interval else None
                
                if len(all_scores[0]) >= ep_max:
                    agent.save_models(epoch) if (epoch+1)%save_interval else None
                    bot.send_string("Training Complete!")
                    capture(agent, game, frame_repeat, action_space, nav_action_space, extract_frame, res, 10, reward_tracker=reward_tracker)
                    game.close()
                    return
                
                sleep(1) # pause for 30 seconds to recover from heat
        except Exception as e:
            bot.send_error(e)
            game.close()
            return
    
    bot.send_string("Training Complete!")        
    game.close()
    return (agent, game, all_scores)

