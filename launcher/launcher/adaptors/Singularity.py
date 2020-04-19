from launcher.adaptors.Adaptor import Adaptor
import os
import subprocess


class Singularity(Adaptor):

    @staticmethod
    def arguments(parser):
        parser.add_argument_group(
            'singularity', 'Arguments for the Singularity Adaptor')

    @staticmethod
    def execute(task, args) -> None:
        os.environ["PYTHONUNBUFFERED"] = "true"
        os.environ["OUTDIR"] = "/scratch/{}/results/{}/".format(
            os.environ["USER"],
            args["dataset"]
        )

        # if [-z ${OUTDIR+x}]
        # then
        # export OUTDIR =
        # fi

        # if [-z ${INDIR+x}]
        # then
        # export INDIR = "/scratch/${USER}/datasets/severstal/"
        # fi

        # if [-z ${OUTFOLDER+x}]
        # then
        # export OUTFOLDER = "output"
        # fi

        # if [-z ${COMMAND+x}]
        # then
        # export COMMAND = "train"
        # fi

        # mkdir - p "${OUTDIR}${OUTFOLDER}"

        print("Using output directory ${OUTDIR}${OUTFOLDER}".format(
            os.environ["OUTDIR"], os.environ["OUTFOLDER"]))
        subprocess.check_call(
            " ".join(args["command"]), shell=args["shell"], cwd=args["path"])
