from launcher import Task
import json
import os
import argparse
import pprint


class ConfigureTask(Task):

    name = 'configure'

    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.join(os.path.abspath(args["data_dir"]),
                                     args["dataset"])
        self.output_dir = os.path.join(os.path.abspath(args["output_dir"]),
                                       args["dataset"])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(ConfigureTask.name,
                                           help='Construct the model.')
        command_parser.add_argument(
            "dataset",
            type=str,
            help='the dataset to use when running the command.')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return args["dataset"]

    def execute(self) -> None:
        from segmenter.config import config_from_env
        self.job_config, self.job_hash = config_from_env(self.data_dir)
        pprint.pprint(self.job_config)
        print(self.job_hash)
        os.makedirs(self.output_dir, exist_ok=True)
        config_location = os.path.join(self.output_dir, self.job_hash,
                                       "config.json")
        with open(config_location, "w") as config_file:
            json.dump(self.job_config, config_file)


tasks = [ConfigureTask]