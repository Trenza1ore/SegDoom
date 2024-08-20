from iqn.iqn import IQN
from iqn.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

__all__ = ["IQN", "CnnPolicy", "MlpPolicy", "MultiInputPolicy"]

# Note: IQN code taken from feat/iqn branch of Stable-Baselines3 - Contrib's Github repo:
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/feat/iqn 
# Slight modifications made to support current version of gymnasium and stable baselines3