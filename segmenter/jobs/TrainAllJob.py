from .BaseJob import BaseJob
import json
import os
import pprint
import itertools
import sys
from launcher.adaptors import Adaptors


class TrainAllJob(BaseJob):

    name = 'train-all'

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([args["dataset"]])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(TrainAllJob.name,
                                           help='Train All Jobs Sequentially.')
        command_parser.add_argument(
            "--train-adaptor",
            type=str,
            default="shell",
            choices=Adaptors.choices(),
            help='the adaptor with which to run the task.')
        BaseJob.arguments(command_parser)

    def execute(self):
        from segmenter.jobs import IsCompleteJob
        is_complete_job = IsCompleteJob(self.args)
        print("Collecting incomplete jobs.")
        _, incomplete = is_complete_job.calculate()
        print("Found %s incomplete jobs" % len(incomplete))

        for i in incomplete:
            cmd = "launch --adaptor %s train severstal --job-config %s --classes %s --folds %s" % (
                self.args["train_adaptor"], i[0], i[1], i[2])
            print("Executing %s" % cmd)
            exit_code = os.system(cmd)

            if exit_code:
                sys.exit(exit_code)
