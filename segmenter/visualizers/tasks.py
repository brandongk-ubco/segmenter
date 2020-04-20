import argparse
from typing import Dict
import pprint
import os
import sys
from segmenter.config import validate_config
from segmenter.aggregators import get_aggregators
from launcher import Task
from segmenter.visualizers import Visualizers
from segmenter.tasks import BaseTask


class VisualizeTask(BaseTask):

    name = 'visualize'

    def __init__(self, args):
        super().__init__(args)
        pprint.pprint(self.job_config)
        validate_config(self.job_config)
        self.visualizer = args["visualizer"].value
        if args["classes"] is not None:
            self.classes = list(
                filter(lambda c: c in args["classes"], self.classes))

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(VisualizeTask.name,
                                           help='Evaluate a model.')
        BaseTask.arguments(command_parser)
        command_parser.add_argument("--visualizer",
                                    type=Visualizers.argparse,
                                    default="auc",
                                    choices=list(Visualizers),
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
        for clazz in self.classes:
            indir = os.path.join(self.output_dir, self.job_hash, clazz,
                                 "results")
            self.visualizer(indir).execute()


tasks = [VisualizeTask]