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
        from segmenter.config import config_from_dir

        super().execute()
        if self.args["classes"] is not None:
            self.classes = list(
                filter(lambda c: c in self.args["classes"], self.classes))

        job_configs = [
            d for d in os.listdir(self.output_dir)
            if os.path.isdir(os.path.join(self.output_dir, d))
        ]

        for job_hash in job_configs:
            job_config, job_hash = config_from_dir(
                os.path.join(self.output_dir, job_hash))

            self.collector(os.path.join(self.output_dir, job_hash),
                           self.data_dir, job_config).execute()


tasks = [CollectTask]