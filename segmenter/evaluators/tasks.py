import argparse
from typing import Dict
import pprint
import os
import sys
from segmenter.config import validate_config
from segmenter.aggregators import get_aggregators
from launcher import Task
from segmenter.evaluators import Evaluators
from segmenter.tasks import BaseTask


class EvaluateTask(BaseTask):

    name = 'evaluate'

    def __init__(self, args):
        super().__init__(args)
        pprint.pprint(self.job_config)
        validate_config(self.job_config)
        self.evaluator = args["evaluator"].value

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(EvaluateTask.name,
                                           help='Evaluate a model.')
        BaseTask.arguments(command_parser)

        command_parser.add_argument("--evaluator",
                                    type=Evaluators.argparse,
                                    default="metric",
                                    choices=list(Evaluators),
                                    help='the evaluation to perform.')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return "--evaluator {} {}".format(args["evaluator"], args["dataset"])

    def execute(self) -> None:
        for aggregator in get_aggregators(self.job_config):
            for clazz in self.classes:
                self.evaluator(clazz, self.job_config, self.job_hash,
                               self.data_dir, self.output_dir,
                               aggregator).execute()


tasks = [EvaluateTask]