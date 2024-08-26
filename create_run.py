import os
import sys
import glob

import clean_up

iswin = sys.platform.startswith("win")
comment = "REM -------new run-------" if iswin else ''
header = "cls\ncall activate doom" if iswin else "clear\n"
tail = "pause" if iswin else ''
script_ext = "bat" if iswin else "sh"

f_replace = lambda content, i: content.replace(
        "task_idx = ", 
        f"task_idx = {i-1}\n# task_idx = "
    ).replace(
        '# DiscordWebhook.send_msg_no_instance(f"Job done: {config}/{name}")', 
        'DiscordWebhook.send_msg_no_instance(f"Job done: {config}/{name}")'
)

with open("train_models.py", encoding="utf-8") as f:
    content = f.read()

if False:
    for i in range(39,40):
        with open(f"train_model_{i}.py", 'w', encoding="utf-8") as f:
            f.write(f_replace(content, i))
        with open(f"train_model_{i}.{script_ext}", 'w', encoding="utf-8") as f:
            f.write(f"{header}\npython train_model_{i}.py\npause")

with open("eval_models.py", encoding="utf-8") as f:
    content = f.read()

# eval_range = list(range(106, 117+1))
eval_range = list(range(141, 165+1))
for i in eval_range:
    with open(f"eval_model_{i}.py", 'w', encoding="utf-8") as f:
        f.write(f_replace(content, i))
    if iswin:
        with open(f"eval_model_{i}.{script_ext}", 'w', encoding="utf-8") as f:
            f.write(f"{header}\npython eval_model_{i}.py\npause")

if True:
    queue = list(range(162, 165+1))
    n_workers = 2
    n_jobs_per_worker = (len(queue) / n_workers).__ceil__()
    i = 0
    special = [] # list(range(72, 63, -1))
    while queue:
        current_jobs = [] + special
        special.clear()
        for _ in [0] * n_jobs_per_worker:
            if queue:
                current_jobs.append(queue.pop())
        # job_section = '\n'.join([f"{comment}\npython eval_model_{j}.py" for j in current_jobs])
        with open(f"eval_model_batch_{i}.py", 'w', encoding="utf-8") as f:
            f.write(content.replace("[task_idx:task_idx+1]", f"[{min(current_jobs)-1}:{max(current_jobs)}]").replace(
                '# DiscordWebhook.send_msg_no_instance(f"Job done: {config}/{name}")',
                'DiscordWebhook.send_msg_no_instance(f"Job done: {config}/{name}"); from random import randrange; sleep(randrange(60, 301))'
            ))
        with open(f"eval_model_batch_{i}.{script_ext}", 'w', encoding="utf-8") as f:
            f.write(f"{header}\npython eval_model_batch_{i}.py\npause")
        i += 1
        print(f"Worker {i}: {sorted(current_jobs)}")

# with open("logs/savefig.py", encoding="utf-8") as f:
#     content = f.read()
# with open(f"plt.{script_ext}", 'w', encoding="utf-8") as f:
#     f.write("\ncls\npython logs/savefig.py\npause")

# for i in [2, 3, 4]:
#     with open(f"plt{i}.{script_ext}", 'w', encoding="utf-8") as f:
#         f.write(f"\ncls\npython logs/savefig{i}.py\npause")

'''
conda activate doombot
eval_model__batch
'''