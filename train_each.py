#!/usr/bin/env python

import subprocess

for clazz in ["1", "2", "3", "4"]:
  subprocess.check_call("TRAIN_CLASS=%s TRAIN_FOLD=0 sbatch job.sh" % clazz, shell=True)
