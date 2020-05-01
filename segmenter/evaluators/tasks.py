import argparse
from launcher import Task
from segmenter.evaluators import Evaluators
from segmenter.tasks import BaseTask
from segmenter.models import FoldWeightFinders


class EvaluateTask(BaseTask):

    name = 'evaluate'

    def __init__(self, args):
        super().__init__(args)
        self.evaluator = Evaluators.get(args["evaluator"])
        self.weight_finder = FoldWeightFinders.get(args["weight_finder"])

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(EvaluateTask.name,
                                           help='Evaluate a model.')
        BaseTask.arguments(command_parser)

        command_parser.add_argument("--evaluator",
                                    type=str,
                                    default="metric",
                                    choices=Evaluators.choices(),
                                    help='the evaluation to perform.')
        command_parser.add_argument("--classes",
                                    type=str,
                                    help='the clases to train',
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
            args["dataset"], "--evaluator {}".format(args["evaluator"]),
            "--weight-finder {}".format(args["weight_finder"]),
            "--classes {}".format(" ".join(args["classes"]))
            if args["classes"] is not None else ""
        ])

    def execute(self) -> None:
        super(EvaluateTask, self).execute()
        if self.args["classes"] is not None:
            self.classes = list(
                filter(lambda c: c in self.args["classes"], self.classes))
        for clazz in self.classes:
            self.evaluator(clazz, self.job_config, self.job_hash,
                           self.data_dir, self.output_dir,
                           self.weight_finder).execute()


tasks = [EvaluateTask]
