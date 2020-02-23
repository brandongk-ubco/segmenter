#!/usr/bin/env python

import subprocess
import os
from args import get_args

limitSeconds = int(os.environ.get("LIMIT_SECONDS", "3600"))

for clazz in ["1", "2", "3", "4"]:
  subprocess.check_call("CLASS=%s LIMIT_SECONDS=%s sbatch %s run.sh" % (clazz, limitSeconds, get_args()), shell=True)
