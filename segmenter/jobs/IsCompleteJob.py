from .BaseJob import BaseJob
import json
import os
import pprint
import itertools
import sys
import glob


class IsCompleteJob(BaseJob):

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
            IsCompleteJob.name, help='Check if training is complete.')
        BaseJob.arguments(command_parser)

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
            if not IsCompleteJob.check_is_complete(root):
                incomplete.append((self.job_hash, clazz, fold))
                continue
            complete.append((self.job_hash, clazz, fold))

        for clazz in self.classes:
            for fold in self.folds:
                directory_exists = os.path.exists(
                    os.path.join(directory, clazz, fold))
                early_stopping_exists = os.path.exists(
                    os.path.join(directory, clazz, fold,
                                 "early_stopping.json"))
                if not directory_exists or not early_stopping_exists:
                    incomplete.append((self.job_hash, clazz, fold))

        complete = sorted(complete, key=lambda x: x[1] + x[2])
        incomplete = sorted(incomplete, key=lambda x: x[1] + x[2])
        return complete, incomplete

    def collect_results(self, directory):
        return [
            os.path.dirname(g)
            for g in glob.glob("{}/**/config.json".format(directory),
                               recursive=True)
        ]

    def _calculate(self, config):
        from segmenter.config import config_from_dir, validate_config

        self.job_config, self.job_hash = config_from_dir(
            os.path.join(self.output_dir, config))
        validate_config(self.job_config)

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

        if self.job_config["SEARCH"]:
            self.folds = ["fold0"]

        return self.is_complete(os.path.join(self.output_dir, config))

    def calculate(self):
        from segmenter.helpers.p_tqdm import t_map as mapper

        if os.environ.get("JOB_CONFIG") is None:
            configs = self.collect_results(self.output_dir)
        else:
            configs = [os.environ["JOB_CONFIG"]]

        complete = []
        incomplete = []
        for job_complete, job_incomplete in mapper(self._calculate, configs):
            complete += job_complete
            incomplete += job_incomplete

        complete = sorted(complete, key=lambda x: x[1] + x[2])
        incomplete = sorted(incomplete, key=lambda x: x[1] + x[2])
        return complete, incomplete

    def execute(self):
        complete, incomplete = self.calculate()
        pprint.pprint({"Complete": complete, "Incomplete": incomplete})
        if len(incomplete) > 0:
            sys.exit(1)
