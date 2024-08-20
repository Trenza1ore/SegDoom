from functools import partial

import numpy as np

from vizdoom.vizdoom import GameVariable

NUM_WEAPON_TYPES:   int = 10

AMMO_VARS:          list[int] = [eval(f"GameVariable.AMMO{i}") for i in range(NUM_WEAPON_TYPES)]
WEAPON_VARS:        list[int] = [eval(f"GameVariable.WEAPON{i}") for i in range(NUM_WEAPON_TYPES)]

HEALTH:             int = GameVariable.HEALTH
FRAG_COUNT:         int = GameVariable.FRAGCOUNT
DAMAGE:             int = GameVariable.DAMAGECOUNT
ARMOR:              int = GameVariable.ARMOR
POS_X:              int = GameVariable.POSITION_X
POS_Y:              int = GameVariable.POSITION_Y

class RewardTracker:
    def __init__(self, game, **kwargs) -> None:
        self.game = game
        self.factor_frag = 1
        self.factor_death = 1
        self.factor_pos_health = 0.02
        self.factor_neg_health = 0.01
        self.factor_pos_ammo = 0.02
        self.factor_neg_ammo = 0.01
        self.factor_damage = 0.01
        self.factor_pos_movement = 0.00005
        self.factor_neg_movement = 0.0025
        self.factor_pos_threshold = 3.0
        for k, v in kwargs.items():
            if k[:7] != "factor_":
                k = "factor_" + k
            setattr(self, k, v)
        self.reset_last_vars()
    
    def reset_last_vars(self) -> None:
        self.last_health = max(self.game.get_game_variable(HEALTH), 0)
        self.last_frag = 0
        self.last_damage = 0
        # self.last_armor = self.game.get_game_variable(ARMOR)
        self.last_pos = np.asfarray([self.game.get_game_variable(POS_X), self.game.get_game_variable(POS_Y)])
        self.last_ammo = np.array([self.game.get_game_variable(a_type) for a_type in AMMO_VARS], dtype=np.int64)
        self.delta_frag = 0
        # self.ep_len = 0
        self.total_reward = 0
    
    def update(self) -> float:
        health = max(self.game.get_game_variable(HEALTH), 0)
        frag = self.game.get_game_variable(FRAG_COUNT)
        damage = self.game.get_game_variable(DAMAGE)
        # armor = self.game.get_game_variable(ARMOR)
        pos = np.asfarray([self.game.get_game_variable(POS_X), self.game.get_game_variable(POS_Y)])
        ammo = np.array([self.game.get_game_variable(a_type) for a_type in AMMO_VARS], dtype=np.int64)

        dhealth = health - self.last_health
        self.delta_frag = frag - self.last_frag
        ddamage = damage - self.last_damage
        # darmor = armor - self.last_armor
        dpos = np.linalg.norm(pos - self.last_pos)
        dammo = ammo - self.last_ammo
        # print(ammo, self.last_ammo, dammo)

        self.last_health = health
        self.last_frag = frag
        self.last_damage = damage
        # self.last_armor = armor
        self.last_pos = pos
        self.last_ammo = ammo

        reward = (dhealth * self.factor_pos_health) if dhealth > 0 else (dhealth * self.factor_neg_health)
        reward += self.delta_frag * self.factor_frag
        reward += ddamage * self.factor_damage
        # reward += darmor
        reward += self.factor_pos_movement if dpos > self.factor_pos_threshold else -self.factor_neg_movement
        reward += dammo[dammo > 0].sum() * self.factor_pos_ammo + dammo[dammo < 0].sum() * self.factor_neg_ammo
        self.total_reward += reward
        # self.ep_len += 1

        return reward
    
    def death_penalty(self) -> int | float:
        return self.factor_death
    
    def get_frag_count(self) -> int | float:
        return self.game.get_game_variable(FRAG_COUNT)