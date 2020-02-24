#!/usr/bin/env python

import subprocess
import os

limitSeconds = int(os.environ.get("LIMIT_SECONDS", "3600"))

for clazz in ["1", "2", "3", "4"]:
  subprocess.check_call("CLASS=%s LIMIT_SECONDS=%s sbatch run.sh" % (clazz, limitSeconds), shell=True)
