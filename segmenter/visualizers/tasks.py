import argparse
from typing import Dict
import pprint
import os
import sys
from launcher import Task
from segmenter.visualizers import Visualizers
from segmenter.tasks import BaseTask


class VisualizeTask(BaseTask):

    name = 'visualize'

    def __init__(self, args):
        super().__init__(args)
        self.visualizer = Visualizers.get(args["visualizer"])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(VisualizeTask.name,
                                           help='Visualize results.')
        BaseTask.arguments(command_parser)
        command_parser.add_argument("--visualizer",
                                    type=str,
                                    default="metric",
                                    choices=Visualizers.choices(),
                                    help='the visualization to perform.')
        command_parser.add_argument("--classes",
                                    type=str,
                                    help='the clases to train',
                                    required=False,
                                    nargs='+')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([
            args["dataset"], "--visualizer {}".format(args["visualizer"]),
            "--classes {}".format(" ".join(args["classes"]))
            if args["classes"] is not None else ""
        ])

    def execute(self) -> None:
        super().execute()
        if self.visualizer.full_combined_visualizer:
            self.visualizer(self.output_dir, self.job_config,
                            self.job_hash).execute()
        elif self.visualizer.job_combined_visualizer:
            indir = os.path.join(self.output_dir, self.job_hash)
            self.visualizer(indir, self.job_config, self.job_hash).execute()
        else:
            if self.args["classes"] is not None:
                self.classes = list(
                    filter(lambda c: c in self.args["classes"], self.classes))
            for clazz in self.classes:
                indir = os.path.join(self.output_dir, self.job_hash, clazz,
                                     "results")
                self.visualizer(indir, self.job_config,
                                self.job_hash).execute()


tasks = [VisualizeTask]
