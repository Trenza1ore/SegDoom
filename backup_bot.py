import os
import sys
import glob
import time

import numpy as np

from stats import DiscordWebhook, ewa_smooth

# Configurations
row_sep = "\n      "
check_interval = 60
also_check_directory = False
also_check_memory_usage = True
memory_usage_check_freq = 5
glob_pattern = "logs/*/evaluations.npz"
models_per_row = 6

# Custom filtering / formatting settings
f_filter_path = lambda k: any(i in k for i in ['sl_ss', 's4_ppo_ss_rgb'])
f_filter_model_names = lambda k: k
f_name_model = lambda k: k.replace("_ss", '').replace("_rgb", '').replace("r_ppo_", '').replace("ppo_", '').replace("Dsl", 'SL').replace("Dsm", 'SM')

# Initialize variables
mem_usage = ''
prev_progress = ''
bot = DiscordWebhook(extra="Backup")
dirs = [f for f in glob.glob(glob_pattern) if f_filter_path(f)]

if __name__ == "__main__":
    os.system("cls" if sys.platform.lower()[:3] == "win" else "clear")

    if also_check_memory_usage:
        import psutil
    else:
        psutil = None
    
    counter = 0

    while True:
        counter += 1
        if also_check_directory:
            dirs = [f for f in glob.glob(glob_pattern) if f_filter_path(f)]

        all_evals_tmp = {f.split(os.path.sep)[-2] : np.load(f) for f in dirs}
        all_evals_tmp = {f_name_model(k) : (v["results"], v["timesteps"]) 
                         for k, v in all_evals_tmp.items() if f_filter_model_names(k)}
        evals_values = list(all_evals_tmp.values())

        all_eval_counts_raw = f"{'prog':4s}: " + ' '.join([f"{len(v):4d}" for v, _ in evals_values])
        all_eval_counts = all_eval_counts_raw

        if all_eval_counts != prev_progress:
            mem_usage = f"{psutil.virtual_memory().available / 1e9:.2f} GB available" if also_check_memory_usage else ''
            ewma = [ewa_smooth(np.mean(v, axis=1), 0.9) for v, _ in evals_values]
            if len(all_evals_tmp) <= models_per_row:
                all_means = f"{'mean':4s}: " + ' '.join([f"{np.mean(v[-1, :]):4.1f}" for v, _ in evals_values])
                all_means_prev = f"{'prev':4s}: " + ' '.join([f"{np.mean(v[-2, :]) if len(v) > 1 else -1:4.1f}" for v, _ in evals_values])
                all_means_best = f"{'best':4s}: " + ' '.join([f"{np.mean(v, axis=1).max():4.1f}" for v, _ in evals_values])
                all_means_ewma = f"{'ewma':4s}: " + ' '.join([f"{v[-1]:4.1f}" for v in ewma])
                all_means_ewma_max = f"{'smax':4s}: " + ' '.join([f"{v.max():4.1f}" for v in ewma])
                all_eval_length = f"{'step':4s}: " + ' '.join([f"{round(v[-1]/1e3):3d}K" if v[-1] < 1e6 else f"{v[-1]/1e6:3.1f}M" for _, v in evals_values])
            else:
                all_eval_counts = f"{'prog':4s}: "
                all_means = f"{'mean':4s}: "
                all_means_prev = f"{'prev':4s}: "
                all_means_best = f"{'best':4s}: "
                all_means_ewma = f"{'ewma':4s}: "
                all_means_ewma_max = f"{'smax':4s}: "
                all_eval_length = f"{'step':4s}: "

                n_rows = (len(all_evals_tmp) / models_per_row).__ceil__()
                for i in range(n_rows):
                    idx_s = i * models_per_row
                    idx_e = idx_s + models_per_row
                    evals_slice = evals_values[idx_s:idx_e]
                    ewma_slice = ewma[idx_s:idx_e]
                    all_eval_counts += ' '.join([f"{len(v):4d}" for v, _ in evals_slice])
                    all_means += ' '.join([f"{np.mean(v[-1, :]):4.1f}" for v, _ in evals_slice])
                    all_means_prev += ' '.join([f"{np.mean(v[-2, :]) if len(v) > 1 else -1:4.1f}" for v, _ in evals_slice])
                    all_means_best += ' '.join([f"{np.mean(v, axis=1).max():4.1f}" for v, _ in evals_slice])
                    all_means_ewma += ' '.join([f"{v[-1]:4.1f}" for v in ewma_slice])
                    all_means_ewma_max += ' '.join([f"{v.max():4.1f}" for v in ewma_slice])
                    all_eval_length += ' '.join([f"{round(v[-1]/1e3):3d}K" if v[-1] < 1e6 else f"{v[-1]/1e6:3.1f}M" for _, v in evals_slice])
                    if i < (n_rows - 1):
                        all_eval_counts += row_sep
                        all_means += row_sep
                        all_means_prev += row_sep
                        all_means_best += row_sep
                        all_means_ewma += row_sep
                        all_means_ewma_max += row_sep
                        all_eval_length += row_sep

            msg = '\n'.join([all_eval_length, all_eval_counts, all_means_prev, all_means, all_means_best, all_means_ewma, all_means_ewma_max, '-'*35] + [f"{k:10s}:" + ' '.join([f"{np.percentile(v[-1, :], i):4.1f}" for i in (0, 25, 50, 75, 100)]) for k, (v, _) in all_evals_tmp.items()])
            update_time = time.strftime("%H:%M")
            bot.send_msg(f"```{update_time:5s} {mem_usage}\n{'-'*35}\n{msg}\n```")
        prev_progress = all_eval_counts_raw

        if also_check_memory_usage and counter >= memory_usage_check_freq:
            counter = 0
            mem_usage = f"{psutil.virtual_memory().available / 1e9:6.2f} GB available"
        print(f"Last sent update: {update_time} (checked at {time.strftime('%H:%M:%S')}) {mem_usage}", end="\r")
        time.sleep(check_interval)