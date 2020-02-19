import subprocess

for clazz in ["1", "2", "3", "4"]:
  subprocess.check_call("TRAIN_CLASS=%s sbatch job.sh" % clazz, shell=True)
