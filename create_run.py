import os
import sys
import glob

import clean_up

iswin = sys.platform.startswith("win")
comment = "REM -------new run-------" if iswin else ''
header = "cls\ncall activate doom" if iswin else "clear\n"
tail = "pause" if iswin else ''
script_ext = "bat" if iswin else "sh"

with open("run.py") as f:
    content = f.read()
with open(f"run.{script_ext}", 'w') as f:
    f.write("f{header}\npython run.py\npause")

# for i in range(39,40):
#     with open(f"run{i}.py", 'w') as f:
#         f.write(content.replace("task_idx = ", f"task_idx = {i-1}\n# task_idx = "))
    # with open(f"run{i}.{script_ext}", 'w') as f:
    #     f.write(f"{header}\npython run{i}.py\npause")

with open("collect.py") as f:
    content = f.read()
# with open(f"collect.{script_ext}", 'w') as f:
#     f.write(f"{header}\npython collect.py\npause")

# eval_range = list(range(106, 117+1))
eval_range = list(range(94, 96+1))
for i in eval_range:
    with open(f"collect{i}.py", 'w') as f:
        f.write(content.replace("task_idx = ", f"task_idx = {i-1}\n# task_idx = "))
    if iswin:
        with open(f"collect{i}.{script_ext}", 'w') as f:
            f.write(f"{header}\npython collect{i}.py\npause")

if False:
    queue = eval_range
    n_workers = 4
    n_jobs_per_worker = (len(queue) / n_workers).__ceil__()
    i = 0
    special = [] # list(range(72, 63, -1))
    while queue:
        current_jobs = [] + special
        special.clear()
        for _ in [0] * n_jobs_per_worker:
            if queue:
                current_jobs.append(queue.pop())
        # job_section = '\n'.join([f"{comment}\npython collect{j}.py" for j in current_jobs])
        with open(f"collect_batch{i}.py", 'w') as f:
            f.write(content.replace("[task_idx:task_idx+1]", f"[{min(current_jobs)-1}:{max(current_jobs)}]"))
        with open(f"collect_batch{i}.{script_ext}", 'w') as f:
            f.write(f"{header}\npython collect_batch{i}.py\npause")
        i += 1
        print(f"Worker {i}: {current_jobs}")

# with open("logs/savefig.py") as f:
#     content = f.read()
# with open(f"plt.{script_ext}", 'w') as f:
#     f.write("\ncls\npython logs/savefig.py\npause")

# for i in [2, 3, 4]:
#     with open(f"plt{i}.{script_ext}", 'w') as f:
#         f.write(f"\ncls\npython logs/savefig{i}.py\npause")

'''
conda activate doombot
collect_batch
'''