import subprocess
import os

folds = int(os.environ.get("FOLDS", 10))

for clazz in ["1", "2", "3", "4"]:
    for fold in range(folds):
      subprocess.check_call("TRAIN_CLASS=%s TRAIN_FOLD=%s sbatch job.sh" % (clazz, fold), shell=True)
