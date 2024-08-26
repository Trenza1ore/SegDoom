import warnings

# This is purely for backward-compatibility with older pre-trained models 
from models.feat_extract import *

msg = "Note: models.DQN is an alias for models.feat_extract...\n"
msg += "Sorry, this is a sketchy backward-compatibility thing...\n"
msg += "I was lazy and put feature extractor CNN in a models/DQN.py"

warnings.warn(msg, FutureWarning)