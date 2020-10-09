from launcher import Task
import argparse
from typing import Dict
import os
import itertools
import sys
import pprint
import json


class BaseJob(Task):
    @staticmethod
    def arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "dataset",
            type=str,
            help='the dataset to use when running the command.',
            nargs='?',
            default="")
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
        self.job_config = None
        self.job_hash = None

        with open(os.path.join(self.data_dir, "classes.json"),
                  "r") as json_file:
            data = json.load(json_file)
            self.classes = data["class_order"]

        if "JOB_CONFIG" in os.environ:
            self.job_config, self.job_hash = config_from_dir(
                os.path.join(self.output_dir, os.environ["JOB_CONFIG"]))
            # validate_config(self.job_config)
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

        try:
            import tensorflow as tf
            if os.environ.get("DEBUG", "false").lower() == "true":
                tf.config.experimental_run_functions_eagerly(True)
            else:
                tf.get_logger().setLevel("ERROR")
        except ModuleNotFoundError:
            pass
