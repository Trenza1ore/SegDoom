import os
import glob

from tqdm.rich import tqdm

to_delete = glob.glob("run?*.*") + glob.glob("collect?*.*") + glob.glob("train_model_?*.*") + glob.glob("eval_model_?*.*")
if to_delete:
    print(f"Delete the following file?\n- {'\n- '.join(sorted(to_delete))}")
    if len(input("Press ENTER directly to delete: ")) == 0:
        for f in tqdm(to_delete):
            os.remove(f)
    else:
        print("Deletion cancelled")
else:
    print("Nothing to clean")