import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from numpy.random import default_rng

from models.DQN import DQNv1, DRQNv2
from models.replay_memory import ReplayMemory
from vizdoom_utils import *
from time import time

# template string for saving path of models and replay memory
model_savepath = "pretrained/%s-model-doom-%d.pth"

if not os.path.exists("pretrained"):
    os.mkdir("pretrained")

class CombatAgent:
    '''My RL agent
    '''
    def __init__(self, device: torch.device, mem_size: int, 
                 action_num: int, nav_action_num: int,
                 discount: float, lr: float, dropout, loss=nn.MSELoss, 
                 act_wd: float=0, nav_wd: float=0, optimizer=optim.SGD, 
                 state_len: int=10, act_model=DRQNv2, nav_model=DQNv1,
                 eps: float=1, eps_decay: float=0.99, eps_min: float=0.1,
                 nav_req_feature: bool=False, ch_num: int=3, name: str='', seed=int(time())):
        
        # store model hyper parameters
        self.device = device
        self.action_num = action_num
        self.nav_action_num = nav_action_num
        self.discount = discount
        self.lr = lr
        self.act_wd = act_wd
        self.nav_wd = nav_wd
        self.optimiser = optimizer
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.name = name
        self.ch_num = ch_num
        
        # set up random number generator
        self.rng = default_rng(seed)
        
        self.history_len = state_len - 2
        self.padding_len = state_len >> 1
        
        # set up models
        self.criterion = loss()
        self.act_net = torch.jit.script(act_model(action_num=action_num, feature_num=1, dropout=dropout, ch_num=ch_num).to(device))
        self.act_net.train()
        if nav_model == None:
            self.no_nav = True
            self.train = self.train_no_nav
        else:
            self.no_nav = False
            self.nav_net = torch.jit.script(nav_model(action_num=nav_action_num, dropout=dropout, ch_num=ch_num).to(device))
            self.nav_net.train()
        
        # set up optimizer (a dict type lr means PPO is used) (PPO support has been dropped)
        if type(lr) == dict:
            self.act_optimizer = optimizer(self.act_net.parameters(), **lr)
            if not self.no_nav:
                self.nav_optimizer = optimizer(self.nav_net.parameters(), **lr)
        else:
            self.act_optimizer = optimizer(self.act_net.parameters(), 
                                        lr=lr, weight_decay=act_wd)
            if not self.no_nav:
                self.nav_optimizer = optimizer(self.nav_net.parameters(), 
                                        lr=lr, weight_decay=nav_wd)
        
        # set up memory for priorotized experience replay
        dtypes = [np.uint8, np.float16, '?', '?', 'f2', 'f2', 'f2']
        # dtypes.append('bool') if nav_req_feature else None
        self.memory = ReplayMemory(res=(144, 256), ch_num=ch_num, size=mem_size, 
                                   history_len=state_len-2, dtypes=dtypes)
    
    def decide_move(self, state: np.ndarray, is_combat: bool) -> torch.tensor:
        state = torch.from_numpy(state.transpose(2,0,1)).float().cuda()
        if self.rng.uniform() < self.eps:
            return self.rng.integers(0, self.action_num) if is_combat else self.rng.integers(0, self.nav_action_num)
        else:
            if is_combat:
                self.act_net.inf_feature(state)
                return torch.argmax(self.act_net.inf_action()).item()
            else:
                return torch.argmax(self.nav_net(state)).item()

    def decide_move_blind(self, state: np.ndarray) -> tuple[torch.tensor, bool]:
        state = torch.from_numpy(state.transpose(2,0,1)).float().cuda()
        if self.act_net.inf_feature(state) > 0.5:
            return (torch.argmax(self.act_net.inf_action()).item(), True)
        else:
            return (torch.argmax(self.nav_net(state)).item(), False)
    
    def eval(self, mode: bool=True):
        '''Sets the models to evaluation mode
        '''
        if mode:
            self.act_net.eval()
            None if self.no_nav else self.nav_net.eval()
        else:
            self.act_net.train()
            None if self.no_nav else self.nav_net.train()
    
    def add_mem(self, state: np.ndarray, action: int, reward: float, features: tuple[bool]):
        self.memory.add(state, reward, action, features)
    
    def add_mem_bulk_unsafe(self, state: np.ndarray, action: np.ndarray[int], reward: np.ndarray[float], 
                            features: np.ndarray[bool]):
        self.memory.bulk_add_unsafe(state, reward, action, features)
    
    def add_mem_bulk(self, state: np.ndarray, action: np.ndarray[int], reward: np.ndarray[float], 
                            features: np.ndarray[bool]):
        self.memory.bulk_add(state, reward, action, features)
    
    def save_models(self, epoch: int) -> str:
        current = dict()
        current["memory"] = self.memory
        current["stat_dict_act"] = self.act_net.state_dict()
        current["stat_dict_opt_act"] = self.act_optimizer.state_dict()
        if not self.no_nav:
            current["stat_dict_nav"] = self.nav_net.state_dict()
            current["stat_dict_opt_nav"] = self.nav_optimizer.state_dict()
        current["hyper_param"] = (self.action_num, self.nav_action_num, 
                                  self.history_len, self.padding_len,
                                  self.discount, self.act_wd, self.nav_wd,
                                  self.eps, self.eps_decay, self.eps_min)
        file_name = model_savepath %(self.name, epoch)
        torch.save(current, file_name, pickle_protocol=5)
        
        del current
        
        return file_name
        
    def load_models(self, epoch: int, name: str='', inference: bool=True):
        self.name = name
        current = torch.load(model_savepath %(name, epoch))
        self.act_net.load_state_dict(current["stat_dict_act"])
        if not self.no_nav:
            self.nav_net.load_state_dict(current["stat_dict_nav"])
        self.action_num, self.nav_action_num, self.history_len, self.padding_len, \
            self.discount, self.act_wd, self.nav_wd, self.eps, self.eps_decay, \
            self.eps_min = current["hyper_param"]
            
        if not inference:
            self.act_optimizer.load_state_dict(current["stat_dict_opt_act"])
            if not self.no_nav:
                self.nav_optimizer.load_state_dict(current["stat_dict_opt_nav"])
            self.memory = current["memory"]
        else:
            self.act_net.eval()
            if not self.no_nav:
                self.nav_net.eval()
        
        # Avoid CUDA out of memory error when model memory allocation is close to 100% vram
        del current
        torch.cuda.empty_cache()
    
    def train(self, batch_size: int=1, feature_loss_factor: float=0.1):
        indices = self.memory.replay_p(batch_size)
        for i in indices:
            start, end  = i-self.history_len, i+2
            frames      = self.memory.frames[start:end, :, :, :]
            rewards     = self.memory.rewards[start:end]
            actions     = self.memory.actions[start:end]
            is_combat   = self.memory.features['0'][start:end]
            ongoing_mask= ~self.memory.features['1'][start:end]
            for s_start in range(0, self.padding_len):
                # We are updating the state (s_end-1) here
                # s_start to s_end-2 are the observation history
                # s_end-1 is the current state
                # s_end is the next state
                s_end = s_start + self.padding_len
                current = s_end-1
                
                is_combat_state = is_combat[current]
                if is_combat_state != is_combat[s_end]:
                    continue
                
                current_action = actions[current]
                
                # Distinguish between combat and navigation task
                if is_combat_state:
                    model   = self.act_net
                    feature = is_combat.copy()
                    optimizer = self.act_optimizer
                    
                    # Predict game features and action values for the next state
                    with torch.no_grad():
                        next_state = torch.from_numpy(frames[s_start+1:s_end+1, :, :, :]).float().cuda()
                        pred_feature = model.inf_feature(next_state)[:, 0]
                        pred_action = torch.max(model.inf_action()[-1]).item()
                    
                    # Calculate loss for game features
                    true_feature = torch.from_numpy(feature[s_start+1:s_end+1]).float().cuda()
                    game_feature_loss = self.criterion(true_feature, pred_feature) * feature_loss_factor
                    
                else:
                    model = self.nav_net
                    optimizer = self.nav_optimizer
                    model2 = self.act_net
                    feature = is_combat.copy()
                
                    # Predict game features and action values for the next state
                    with torch.no_grad():
                        next_state = torch.from_numpy(frames[s_end, :, :, :]).float().cuda()
                        pred_action = torch.max(model(next_state)).item()
                    
                    next_state2 = torch.from_numpy(frames[s_start+1:s_end+1, :, :, :]).float().cuda()
                    pred_feature = model2.inf_feature(next_state2)[:, 0]
                    
                    # Calculate loss for game features
                    true_feature = torch.from_numpy(feature[s_start+1:s_end+1]).float().cuda()
                    game_feature_loss = self.criterion(true_feature, pred_feature) * feature_loss_factor
                
                # q-target = reward + discount * max_(a')(q(s',a'))
                q_target = torch.tensor(
                    rewards[current] + (self.discount * pred_action) if ongoing_mask[s_end-1] else rewards[current],
                    dtype=torch.float32, device=self.device)
                
                # Predict action values for the current state
                if is_combat_state:
                    model.inf_feature(torch.from_numpy(frames[s_start:s_end, :, :, :]).float().cuda())
                    pred_action = model.inf_action()[-1, current_action]
                    loss = self.criterion(q_target, pred_action)+game_feature_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    pred_action = model(torch.from_numpy(frames[current, :, :, :]).float().cuda())[0, current_action]
                    loss = self.criterion(q_target, pred_action)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.act_optimizer.zero_grad()
                    game_feature_loss.backward()
                    self.act_optimizer.step()
                
        self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min
    
    def nav_train(self, batch_size: int=5):
        indices = self.memory.replay_p_nav(batch_size)
        for i in indices:
            start, end  = i-self.history_len, i+2
            frames      = self.memory.frames[start:end, :, :, :]
            rewards     = self.memory.rewards[start:end]
            actions     = self.memory.actions[start:end]
            is_combat   = self.memory.features['0'][start:end]
            ongoing_mask= ~self.memory.features['1'][start:end]
            
            for s_start in range(0, self.padding_len):
                # We are updating the state (s_end-1) here
                # s_start to s_end-2 are the observation history
                # s_end-1 is the current state
                # s_end is the next state
                s_end = s_start + self.padding_len
                current = s_end-1
                
                # extra protection because numpy's choice based on probability is unreliable
                if is_combat[current] or is_combat[s_end]:
                    continue
                
                current_action = actions[current]
                
                model = self.nav_net
                optimizer = self.nav_optimizer

            
                # Predict game features and action values for the next state
                with torch.no_grad():
                    next_state = torch.from_numpy(frames[s_end, :, :, :]).float().cuda()
                    pred_action = torch.max(model(next_state)).item()
                
                # q-target = reward + discount * max_(a')(q(s',a'))
                q_target = torch.tensor(
                    rewards[current] + (self.discount * pred_action) if ongoing_mask[s_end-1] else rewards[current],
                    dtype=torch.float32, device=self.device)
                
                # Predict action values for the current state
                pred_action = model(torch.from_numpy(frames[current, :, :, :]).float().cuda())[0, current_action]
                
                # Calculate loss for action values
                loss = self.criterion(q_target, pred_action)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    # Legacy code that I still want to keep as reference
    
    # def train_no_nav(self, batch_size: int=5, feature_loss_factor: float=100):
    #     indices = self.memory.replay_p(batch_size)
    #     for i in indices:
    #         start, end  = i-self.history_len, i+2
    #         frames      = self.memory.frames[start:end, :, :, :]
    #         rewards     = self.memory.rewards[start:end]
    #         actions     = self.memory.actions[start:end]
    #         is_combat   = self.memory.features[start:end, 0]
    #         ongoing_mask= ~self.memory.features[start:end, 1]
            
    #         for s_start in range(0, self.padding_len):
    #             # We are updating the state (s_end-1) here
    #             # s_start to s_end-2 are the observation history
    #             # s_end-1 is the current state
    #             # s_end is the next state
    #             s_end = s_start + self.padding_len
    #             current = s_end-1
                
    #             model = self.act_net
    #             feature = is_combat.copy()
    #             optimizer = self.act_optimizer
                
    #             current_action = actions[current]
                
    #             # Predict game features and action values for the next state
    #             with torch.no_grad():
    #                 next_state = torch.from_numpy(frames[s_start+1:s_end+1, :, :, :]).float().cuda()
    #                 pred_feature = model.inf_feature(next_state)[:, 0]
    #                 pred_action = torch.max(model.inf_action()[-1]).item()
                
    #             # Calculate loss for game features
    #             true_feature = torch.from_numpy(feature[s_start+1:s_end+1]).float().cuda()
    #             game_feature_loss = self.criterion(true_feature, pred_feature) * feature_loss_factor
                
    #             # q-target = reward + discount * max_(a')(q(s',a'))
    #             q_target = torch.tensor(
    #                 rewards[current] + (self.discount * pred_action) if ongoing_mask[s_end-1] else rewards[current],
    #                 dtype=torch.float32, device=self.device)
                
    #             # Predict action values for the current state
    #             model.inf_feature(torch.from_numpy(frames[s_start:s_end, :, :, :]).float().cuda())
    #             pred_action = model.inf_action()[-1, current_action]
                
    #             # Calculate loss for action values
    #             action_value_loss = self.criterion(q_target, pred_action)
    #             optimizer.zero_grad()
    #             (action_value_loss + game_feature_loss).backward()
    #             optimizer.step()
    #     self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min
    
    # def decide_move_nav(self, state: np.ndarray) -> torch.tensor:
    #     if self.rng.uniform() < self.eps:
    #         return self.rng.integers(0, self.nav_action_num)
    #     else:
    #         state = torch.from_numpy(state).float().cuda()
    #         self.nav_net.inf_feature(state)
    #         return torch.argmax(self.nav_net.inf_action()).item()
    
    # def decide_move_no_nav(self, state: np.ndarray) -> torch.tensor:
    #     if self.rng.uniform() < self.eps:
    #         return self.rng.integers(0, self.action_num)
    #     else:
    #         state = torch.from_numpy(state).float().cuda()
    #         self.act_net.inf_feature(state)
    #         return torch.argmax(self.act_net.inf_action()).item()
    
    # def decide_move(self, state: np.ndarray, is_combat: bool) -> torch.tensor:
    #     if self.rng.uniform() < self.eps:
    #         return self.rng.integers(0, self.action_num) if is_combat else self.rng.integers(0, self.nav_action_num)
    #     else:
    #         state = torch.from_numpy(state).float().cuda()
    #         if is_combat:
    #             self.act_net.inf_feature(state)
    #             return torch.argmax(self.act_net.inf_action()).item()
    #         else:
    #             self.nav_net.inf_feature(state)
    #             return torch.argmax(self.nav_net.inf_action()).item()
    
