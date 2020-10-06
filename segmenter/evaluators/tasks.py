import argparse
from launcher import Task
from segmenter.evaluators import Evaluators
from segmenter.jobs import BaseJob
from segmenter.models import FoldWeightFinders
import itertools


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
        return " ".join([
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

    def execute(self) -> None:
        from segmenter.aggregators import Aggregators

        super(EvaluateTask, self).execute()
        if self.args["classes"] is not None:
            self.classes = list(
                filter(lambda c: c in self.args["classes"], self.classes))

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

        if self.args["folds"] is not None:
            self.folds = list(
                filter(lambda c: c in self.args["folds"], self.folds))

        if self.job_config["FOLDS"] == 0:
            self.aggregators = ["dummy"]
        else:
            self.aggregators = Aggregators.choices()

        if self.args["aggregators"] is not None:
            self.aggregators = list(
                filter(lambda c: c in self.args["aggregators"],
                       self.aggregators))

        for clazz in self.classes:
            self.evaluator(clazz,
                           self.job_config,
                           self.job_hash,
                           self.data_dir,
                           self.output_dir,
                           self.weight_finder,
                           folds=self.folds,
                           aggregators=self.aggregators).execute()


tasks = [EvaluateTask]
