# Doom-playing AI via Deep Reinforcement Learning with Semantic Segmentation

**An Informatics MSc Project at University of Edinburgh**
![ss+rgb_rgb](rtss_map1_ep5_rgb.gif)![ss+rgb_rgb](rtss_map1_ep5_ss.gif)
#### Functionality of scripts in project's root directory:
- **run.py -** runs the training session
- **collect.py -** collects data from evalution episodes (game frames, position, etc.)
- **playit.py -** play a scenario yourself
- **tba**
---
# All deprecated information, update in progress...
![](test_resnet101.gif) ![](test_mobilenetv3.gif)
---
Utilizing DQN, DRQN and Prioritized Experience Replay to train an agent for playing Doom. 
Scenarios tested: 
- **Deathmatch (modified):** a modified version of deathmatch scenario with different map layout and texture where pickups are removed and killing enemies restore health/armor/ammo
- **Deadly Corridor (modified):** a more Doom-ish version of deadly corridor scenario where the player starts with a shotgun and no longer takes double damage
- **Deadly Corridor (original):** the classic deadly corridor scenario included in ViZDoom

---

This project was developed using **Python 3.10.9** on a laptop with a CUDA-enabled graphics card. 
Only ~1.6GB of video memory is needed according to my testing, so it should be able to run even on entry-level Nvidia graphics cards like MX 150/GT 1030.
Remove all .cuda() function calls if trained using CPU instead.

**ViZDoom 1.2** now supports **Python 3.11** and **PyTorch 2.0** shouldn't break support, upgrading these might be desirable. The **requirements.txt** provided still uses my current environment as I haven't had the time to test. Support for automatic mixed-precision floating-point tensors can be enabled with **torch.amp**, should speed-up training at the cost of slight reduction in precision, but I haven't had the time to test that either.

---

This agent was trained using ViZDoom: https://github.com/Farama-Foundation/ViZDoom

---

**P.S.** If you want to change the episode timeout setting to > 1050 for any scenario at Nightmare difficulty (skill level 5), note that enemies respawn after 30 seconds (1050 ticks) unless the Thing_Remove function is called in your ACS script to remove them or they die of special ways specifically defined in Doom's source code.
