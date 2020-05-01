import argparse
from typing import Dict
import pprint
import os
from launcher import Task
import itertools
import sys
import json

if os.environ.get("DEBUG", "false").lower() != "true":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BaseTask(Task):
    @staticmethod
    def arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "dataset",
            type=str,
            help='the dataset to use when running the command.')
        parser.add_argument(
            "--job-config",
            type=str,
            help=
            'the configuration on which to run the command. Can also be set through the JOB_CONFIG environment variable.',
            required=False)

    @staticmethod
    def arguments_to_cli(args) -> str:
        return args["dataset"]

    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.join(os.path.abspath(args["data_dir"]),
                                     args["dataset"])
        self.output_dir = os.path.join(os.path.abspath(args["output_dir"]),
                                       args["dataset"])
        if args["job_config"] is not None:
            os.environ["JOB_CONFIG"] = args["job_config"]

    def execute(self):
        from segmenter.config import config_from_dir, validate_config
        self.job_config, self.job_hash = config_from_dir(self.output_dir)
        validate_config(self.job_config)
        pprint.pprint(self.job_config)

        try:
            import tensorflow as tf
            if os.environ.get("DEBUG", "false").lower() == "true":
                tf.config.experimental_run_functions_eagerly(True)
            else:
                tf.get_logger().setLevel("ERROR")
        except ModuleNotFoundError:
            pass

        self.classes = self.job_config["CLASSES"]
        self.folds = ["all"] if self.job_config["FOLDS"] == 0 else [
            "fold{}".format(o) for o in range(self.job_config["FOLDS"])
        ]

        if self.job_config["BOOST_FOLDS"] > 0:
            boost_folds = [
                "b{}".format(o)
                for o in list(range(0, self.job_config["BOOST_FOLDS"] + 1))
            ]
            self.folds = [
                "".join(o)
                for o in itertools.product(*[self.folds, boost_folds])
            ]


class IsCompleteTask(BaseTask):

    name = 'is-complete'

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([
            args["dataset"], "--folds {}".format(" ".join(args["folds"]))
            if args["folds"] is not None else ""
        ])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(
            IsCompleteTask.name, help='Check if training is complete.')
        BaseTask.arguments(command_parser)

    @staticmethod
    def check_is_complete(directory):
        file_path = os.path.join(directory, "early_stopping.json")
        if not os.path.exists(file_path):
            return False
        with open(file_path, "r") as json_file:
            early_stopping = json.load(json_file)
        return early_stopping["wait"] >= early_stopping["patience"]

    def is_complete(self, directory):
        incomplete = []
        complete = []

        # Check for existing directories which are not complete.
        for root, _dirs, _files in os.walk(directory):
            if root.split("/")[-3] != self.job_hash or root.split(
                    "/")[-1] not in self.folds or root.split(
                        "/")[-2] not in self.classes:
                continue
            clazz = root.split("/")[-2]
            fold = root.split("/")[-1]
            if not IsCompleteTask.check_is_complete(root):
                print("WARNING: %s is not complete!" % os.path.abspath(root))
                incomplete.append((self.job_hash, clazz, fold))
                continue
            complete.append((self.job_hash, clazz, fold))

        for clazz in self.classes:
            for fold in self.folds:
                directory_exists = os.path.exists(
                    os.path.join(directory, self.job_hash, clazz, fold))
                early_stopping_exists = os.path.exists(
                    os.path.join(directory, self.job_hash, clazz, fold,
                                 "early_stopping.json"))
                if not directory_exists or not early_stopping_exists:
                    print("WARNING: class {} fold {} is not started!".format(
                        clazz, fold))
                    incomplete.append((self.job_hash, clazz, fold))

        complete = sorted(complete, key=lambda x: x[1] + x[2])
        incomplete = sorted(incomplete, key=lambda x: x[1] + x[2])
        return complete, incomplete

    def execute(self):
        from segmenter.config import config_from_dir, validate_config

        self.job_config, self.job_hash = config_from_dir(self.output_dir)
        validate_config(self.job_config)
        pprint.pprint(self.job_config)

        self.classes = self.job_config["CLASSES"]
        self.folds = ["all"] if self.job_config["FOLDS"] == 0 else [
            "fold{}".format(o) for o in range(self.job_config["FOLDS"])
        ]

        if self.job_config["BOOST_FOLDS"] > 0:
            boost_folds = [
                "b{}".format(o)
                for o in list(range(0, self.job_config["BOOST_FOLDS"] + 1))
            ]
            self.folds = [
                "".join(o)
                for o in itertools.product(*[self.folds, boost_folds])
            ]

        complete, incomplete = self.is_complete(os.path.join(self.output_dir))
        pprint.pprint({"Complete": complete, "Incomplete": incomplete})
        if len(incomplete) > 0:
            sys.exit(1)


class TrainTask(BaseTask):

    name = 'train'

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([
            args["dataset"], "--folds {}".format(" ".join(args["folds"]))
            if args["folds"] is not None else "",
            "--classes {}".format(" ".join(args["classes"]))
            if args["classes"] is not None else ""
        ])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(TrainTask.name,
                                           help='Train a model.')
        BaseTask.arguments(command_parser)
        command_parser.add_argument("--folds",
                                    type=str,
                                    help='the folds to train.',
                                    required=False,
                                    nargs='+')
        command_parser.add_argument("--classes",
                                    type=str,
                                    help='the clases to train',
                                    required=False,
                                    nargs='+')

    def execute(self) -> None:
        from segmenter.train import train_fold
        super().execute()
        if self.args["classes"] is not None:
            self.classes = list(
                filter(lambda c: c in self.args["classes"], self.classes))
        if self.args["folds"] is not None:
            self.folds = list(
                filter(
                    lambda c: c.split("b")[0].replace("fold", "") in self.args[
                        "folds"], self.folds))
        for clazz in self.classes:
            for fold in self.folds:
                train_fold(clazz, fold, self.job_config, self.job_hash,
                           self.data_dir, self.output_dir)


tasks = [TrainTask, IsCompleteTask]
