from .BaseJob import BaseJob


class TrainJob(BaseJob):

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
        command_parser = parser.add_parser(TrainJob.name,
                                           help='Train a model.')
        BaseJob.arguments(command_parser)
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

        if self.job_config["SEARCH"]:
            self.folds = ["fold0"]

        for clazz in self.classes:
            for fold in self.folds:
                train_fold(clazz, fold, self.job_config, self.job_hash,
                           self.data_dir, self.output_dir)
