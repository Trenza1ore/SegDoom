with open("run.py") as f:
    content = f.read()
with open(f"run.bat", 'w') as f:
    f.write("cls\ncall activate doom\npython run.py\npause")

for i in range(2, 21+1):
    with open(f"run{i}.py", 'w') as f:
        f.write(content.replace("task_idx = 0", f"task_idx = {i-1}"))
    with open(f"run{i}.bat", 'w') as f:
        f.write(f"cls\ncall activate doom\npython run{i}.py\npause")

with open("collect.py") as f:
    content = f.read()
with open(f"collect.bat", 'w') as f:
    f.write("cls\ncall activate doom\npython collect.py\npause")

# for i in range(10, 28):
#     with open(f"collect{i}.py", 'w') as f:
#         f.write(content.replace("task_idx = 0", f"task_idx = {i-1}"))
#     with open(f"collect{i}.bat", 'w') as f:
#         f.write(f"cls\ncall activate doom\npython collect{i}.py\npause")

# queue = list(range(16, 28))
# n_workers = 4
# n_jobs_per_worker = (len(queue) / n_workers).__ceil__()
# i = 0
# while queue:
#     current_jobs = []
#     for _ in [0] * n_jobs_per_worker:
#         if queue:
#             current_jobs.append(queue.pop())
#     job_section = '\n'.join([f"REM -------new run-------\npython collect{j}.py" for j in current_jobs])
#     with open(f"collect_batch{i}.bat", 'w') as f:
#         f.write(f"cls\ncall activate doom\n{job_section}\npause")
#     i += 1

# with open("logs/savefig.py") as f:
#     content = f.read()
# with open(f"plt.bat", 'w') as f:
#     f.write("\ncls\npython logs/savefig.py\npause")

# for i in [2, 3, 4]:
#     with open(f"plt{i}.bat", 'w') as f:
#         f.write(f"\ncls\npython logs/savefig{i}.py\npause")