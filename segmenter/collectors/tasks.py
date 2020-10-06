import argparse
from typing import Dict
import pprint
import os
import sys
from launcher import Task
from segmenter.collectors import Collectors
from segmenter.jobs import BaseJob


class CollectTask(BaseJob):

    name = 'collect'

    def __init__(self, args):
        super().__init__(args)
        self.collector = Collectors.get(args["collector"])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(CollectTask.name,
                                           help='Collect results.')
        BaseJob.arguments(command_parser)
        command_parser.add_argument("--collector",
                                    type=str,
                                    default="metric",
                                    choices=Collectors.choices(),
                                    help='the collector to execute.')
        command_parser.add_argument("--classes",
                                    type=str,
                                    help='the clases to train',
                                    required=False,
                                    nargs='+')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join(
            [args["dataset"], "--collector {}".format(args["collector"])])

    def execute(self) -> None:
        super().execute()
        if self.args["classes"] is not None:
            self.classes = list(
                filter(lambda c: c in self.args["classes"], self.classes))
        for clazz in self.classes:
            indir = os.path.join(self.output_dir, self.job_hash, clazz,
                                 "results")
            self.collector(indir, self.data_dir, self.job_config).execute()


tasks = [CollectTask]