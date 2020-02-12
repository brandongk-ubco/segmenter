import subprocess

for clazz in ["1", "2", "3", "4"]:
  for fold in range(5):
    subprocess.check_call("TRAIN_CLASS=%s TRAIN_FOLD=%s sbatch job.sh" % (clazz, fold), shell=True)
