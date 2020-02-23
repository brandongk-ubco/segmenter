
import os
from datetime import timedelta

def get_args(time_buffer=1200):
    args = []

    gpus = int(os.environ.get("GPUS", "1"))
    if gpus > 0:
        args.append("--gres=gpu:%s" % gpus)

    nodes = int(os.environ.get("NODES", "1"))
    args.append("--nodes=%s" % nodes)

    mem = os.environ.get("MEM", "16G")
    args.append("--mem=%s" % mem)

    cpus = int(os.environ.get("CPUS", "2"))
    args.append("--cpus-per-task=%s" % cpus)

    limit = int(os.environ.get("LIMIT_SECONDS", "3600")) + time_buffer
    if limit > 0:
        args.append("--time=%s" % str(timedelta(seconds=limit)))
    
    return " ".join(args)

if __name__ == "__main__":
    print(get_args())