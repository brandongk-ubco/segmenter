import argparse
from launcher import Task
from segmenter.evaluators import Evaluators
from segmenter.jobs import BaseJob
from segmenter.models import FoldWeightFinders
import itertools
import os


class EvaluateTask(BaseJob):

    name = 'evaluate'

    def __init__(self, args):
        super().__init__(args)
        self.evaluator = Evaluators.get(args["evaluator"])
        self.weight_finder = FoldWeightFinders.get(args["weight_finder"])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(EvaluateTask.name,
                                           help='Evaluate a model.')
        BaseJob.arguments(command_parser)

        command_parser.add_argument("--evaluator",
                                    type=str,
                                    default="metric",
                                    choices=Evaluators.choices(),
                                    help='the evaluation to perform.')
        command_parser.add_argument("--classes",
                                    type=str,
                                    help='the clases to evaluate',
                                    required=False,
                                    nargs='+')
        command_parser.add_argument("--folds",
                                    type=str,
                                    help='the folds to evaluate.',
                                    required=False,
                                    nargs='+')
        command_parser.add_argument("--aggregators",
                                    type=str,
                                    help='the aggregators to evaluate.',
                                    required=False,
                                    nargs='+')
        command_parser.add_argument(
            "--weight-finder",
            type=str,
            default="organized",
            choices=FoldWeightFinders.choices(),
            help='the strategy for finding fold weights')

    @staticmethod
    def arguments_to_cli(args) -> str:
        args = " ".join([
            args["dataset"],
            "--evaluator {}".format(args["evaluator"]),
            "--weight-finder {}".format(args["weight_finder"]),
            "--classes {}".format(" ".join(args["classes"]))
            if args["classes"] is not None else "",
            "--folds {}".format(" ".join(args["folds"]))
            if args["folds"] is not None else "",
            "--aggregators {}".format(" ".join(args["aggregators"]))
            if args["aggregators"] is not None else "",
        ])
        return args

    def execute(self) -> None:
        from segmenter.aggregators import Aggregators
        from segmenter.config import config_from_dir
        super(EvaluateTask, self).execute()

        job_configs = [
            d for d in os.listdir(self.output_dir)
            if os.path.isdir(os.path.join(self.output_dir, d))
        ]

        if self.args["classes"] is not None:
            self.classes = list(
                filter(lambda c: c in self.args["classes"], self.classes))

        for job_hash in job_configs:
            job_config, job_hash = config_from_dir(
                os.path.join(self.output_dir, job_hash))

            folds = ["all"] if job_config["FOLDS"] == 0 else [
                "fold{}".format(o) for o in range(job_config["FOLDS"])
            ]
            if job_config["BOOST_FOLDS"] > 0:
                boost_folds = [
                    "b{}".format(o)
                    for o in list(range(0, job_config["BOOST_FOLDS"] + 1))
                ]
                folds = [
                    "".join(o)
                    for o in itertools.product(*[self.folds, boost_folds])
                ]

            if self.args["folds"] is not None:
                folds = list(filter(lambda c: c in self.args["folds"], folds))

            if job_config["SEARCH"]:
                folds = ["fold0"]

            if len(folds) <= 1:
                aggregators = ["dummy"]
            else:
                aggregators = Aggregators.choices()

            if self.args["aggregators"] is not None:
                aggregators = list(
                    filter(lambda c: c in self.args["aggregators"],
                           aggregators))

            for clazz in self.classes:
                self.evaluator(clazz,
                               job_config,
                               job_hash,
                               self.data_dir,
                               self.output_dir,
                               self.weight_finder,
                               folds=folds,
                               aggregators=aggregators).execute()


tasks = [EvaluateTask]
