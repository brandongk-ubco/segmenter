from .BaseJob import BaseJob
import argparse
from typing import Dict
import os
import itertools
import sys
import pprint
from importlib.machinery import SourceFileLoader
from importlib import util
import itertools


class GridSearchJob(BaseJob):

    name = "grid-search"

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(GridSearchJob.name,
                                           help='Configure a grid search.')
        command_parser.add_argument("--config",
                                    type=str,
                                    help='the configuration file to use.',
                                    required=False)
        BaseJob.arguments(command_parser)

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([
            args["dataset"], args["config"],
            "--config {}".format(" ".join(args["config"]))
        ])

    def execute_result(self, config):
        for key, var in zip(self.keys, config):
            os.environ[key] = str(var)
        os.environ["SEARCH"] = "True"
        os.system("launch configure %s  > /dev/null" % self.args["dataset"])

    def execute(self) -> None:
        from segmenter.helpers.p_tqdm import p_map as mapper

        filename = self.args["config"]
        loader = SourceFileLoader("searchconfig", filename)
        spec = util.spec_from_loader("searchconfig", loader)
        searchconfig = util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(searchconfig)  # type: ignore

        self.keys = [k for k in searchconfig.search_space.keys()]
        configs = [
            l for l in itertools.product(*searchconfig.search_space.values())
        ]

        mapper(self.execute_result, configs)
