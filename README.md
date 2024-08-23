# Improving Reinforcement Learning Agents' Performance and Memory Efficiency in 3D Environment via Semantic Segmentation
An Informatics MSc Project at the University of Edinburgh
---
![ss+rgb_ss](rtss_map1_ep5_ss.gif)

**An Informatics MSc Project at University of Edinburgh**

Some gameplay videos are made available at Google Drive:
- [PPO Agent with RGB+SS input (uses DeepLabV3 with ResNet-101 backbone for SS, no frame stack)](https://drive.google.com/drive/folders/17ngSPZ5X83kN_Qn9ufbYl2IgwzxG2dVv?usp=drive_link)

Here's my favourite episode:

https://github.com/user-attachments/assets/c0ce34f6-ae1d-444c-ad2e-5415aa02d453

#### Functionality of scripts in project's root directory:
- **train_models.py -** runs the training session
- **eval_models.py -** collects data from evaluation episodes (game frames, position, etc.)
- **tasks_eval.py -** defines what tasks to be executed by `tasks_eval.py`
- **playit.py -** play a scenario yourself
- **create_run.py -** a sketchy solution to create temporary scripts for train/eval tasks
- **clean_up.py -** cleans temporary scripts
---
# All deprecated information, update in progress...

## Comparison of Semantic Segmentation models
**DeepLabV3 with ResNet-101 backbone:**

https://github.com/user-attachments/assets/f6f2140e-345c-4cb4-ba76-53ba318ffe2e

**DeepLabV3 with MobileNet-V3 backbone:**

https://github.com/user-attachments/assets/459fcbe2-ee09-4c86-b7f5-72e22e6850b1


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
