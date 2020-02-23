#!/usr/bin/env python

import subprocess
import os
from args import get_args

for clazz in ["1", "2", "3", "4"]:
  subprocess.check_call("CLASS=%s sbatch %s run.sh" % (clazz, get_args()), shell=True)
