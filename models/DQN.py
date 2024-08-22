# This is purely for backward-compatibility with older pre-trained models 
from models.feat_extract import *
print("Note: models.DQN is an alias for models.feat_extract...")
print("Sorry, this is a sketchy backward-compatibility thing...")
print("I was lazy and put feature extractor CNN in a models/DQN.py", flush=True)