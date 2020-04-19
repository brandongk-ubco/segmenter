import argparse
from typing import Dict
import pprint
import os
import sys
from segmenter.config import get_config
from segmenter.aggregators import get_aggregators
from launcher import Task
import os
from segmenter.helpers import hash
from segmenter.train import train_fold
from segmenter.evaluators import metric_evaluation
import itertools


class BaseTask(Task):
    @staticmethod
    def arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "dataset",
            type=str,
            help='the dataset to use when running the command.')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return args["dataset"]

    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.join(os.path.abspath(args["data_dir"]),
                                     args["dataset"])
        self.output_dir = os.path.join(os.path.abspath(args["output_dir"]),
                                       args["dataset"])
        try:
            import tensorflow as tf
            if os.environ.get("DEBUG", "false").lower() == "true":
                tf.config.experimental_run_functions_eagerly(True)
            else:
                tf.get_logger().setLevel("ERROR")
        except ModuleNotFoundError:
            pass

        self.job_config = get_config(self.data_dir)
        self.job_hash = hash(self.job_config)
        pprint.pprint(self.job_config)

        self.classes = self.job_config["CLASSES"] if os.environ.get(
            "CLASS") is None else [os.environ.get("CLASS")]

        if self.job_config["FOLDS"] is not None:
            self.folds = ["fold{}".format(os.environ["FOLD"])
                          ] if os.environ.get("FOLD") is not None else [
                              "fold{}".format(o)
                              for o in range(self.job_config["FOLDS"])
                          ]
        else:
            self.folds = ["all"]

        if self.job_config["BOOST_FOLDS"] is not None:
            boost_folds = [
                "b{}".format(o)
                for o in list(range(0, self.job_config["BOOST_FOLDS"] + 1))
            ]
            self.folds = [
                "".join(o)
                for o in itertools.product(*[self.folds, boost_folds])
            ]


class EvaluateTask(BaseTask):

    name = 'evaluate'

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(EvaluateTask.name,
                                           help='Evaluate a model.')
        BaseTask.arguments(command_parser)

    def execute(self) -> None:
        for aggregator in get_aggregators(self.job_config):
            for clazz in self.classes:
                metric_evaluation(clazz, self.job_config, self.job_hash,
                                  self.data_dir, self.output_dir, aggregator)


class TrainTask(BaseTask):

    name = 'train'

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(TrainTask.name,
                                           help='Train a model.')
        BaseTask.arguments(command_parser)

    def execute(self) -> None:
        for clazz in self.classes:
            for fold in self.folds:
                train_fold(clazz, fold, self.job_config, self.job_hash,
                           self.data_dir, self.output_dir)


tasks = [EvaluateTask, TrainTask]
