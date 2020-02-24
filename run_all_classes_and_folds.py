#!/usr/bin/env python

import subprocess
import os

folds = int(os.environ.get("FOLDS", 10))
limitSeconds = int(os.environ.get("LIMIT_SECONDS", "3600"))

for clazz in ["1", "2", "3", "4"]:
    for fold in range(folds):
      subprocess.check_call("CLASS=%s FOLD=%s LIMIT_SECONDS=%s sbatch run.sh" % (clazz, fold, limitSeconds), shell=True)
