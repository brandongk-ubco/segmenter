import argparse
from typing import Dict
import pprint
import os
import sys
from segmenter.config import get_config
from launcher import Task
import os
from segmenter.helpers import hash
from segmenter.train import train_fold


class BaseTask(Task):

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(
            Train.name, help='Train a model.')
        command_parser.add_argument("dataset", type=str,
                                    help='the dataset to use when running the command.')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return args["dataset"]

    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.join(
            os.path.abspath(args["data_dir"]), args["dataset"])
        self.output_dir = os.path.join(
            os.path.abspath(args["output_dir"]), args["dataset"])
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


class Train(BaseTask):

    name = 'train'

    def execute(self) -> None:
        classes = [os.environ.get("CLASS")] if os.environ.get(
            "CLASS") is not None else self.job_config["CLASSES"]
        if self.job_config["FOLDS"] is not None:
            folds = [int(os.environ["FOLD"])] if os.environ.get(
                "FOLD") is not None else range(self.job_config["FOLDS"])
            for clazz in classes:
                for fold in folds:
                    self.train(clazz, fold)
        else:
            for clazz in classes:
                self.train(clazz)

    def train(self, clazz, fold=None):
        if self.job_config["BOOST_FOLDS"] is None:
            train_fold(clazz, self.job_config, self.job_hash, self.data_dir,
                       self.output_dir, fold=fold)
        else:
            for boost_fold in range(0, self.job_config["BOOST_FOLDS"] + 1):
                train_fold(clazz, self.job_config, self.job_hash, self.data_dir,
                           self.output_dir, fold=fold, boost_fold=boost_fold)


tasks = [
    Train
]
