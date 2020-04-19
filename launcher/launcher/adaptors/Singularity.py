from launcher.adaptors.Adaptor import Adaptor
import os
import subprocess


class Singularity(Adaptor):

    @staticmethod
    def arguments(parser):
        parser.add_argument_group(
            'singularity', 'Arguments for the Singularity Adaptor')

    @staticmethod
    def execute(task, args) -> None:
        os.environ["PYTHONUNBUFFERED"] = "true"

        data = {
            "path": os.path.abspath("."),
            "project": os.path.basename(os.path.abspath(".")),
            "data_dir": args["data_dir"],
            "output_dir": args["output_dir"],
            "uid": os.geteuid(),
            "gid": os.getgid(),
            "image": args["image"],
            "command": "launch",
            "args": "{} {}".format(task.name, task.arguments_to_cli(args)),
            "gpus": "--gpus all" if args["gpus"] else ""
        }

        raise NotImplementedError("Not done!")
        # command = "docker run -v {path}/{project}:/src/{project} -v {data_dir}:/data -v {output_dir}:/output -u {uid}:{gid} {gpus} --entrypoint {command} -it {image} {args}".format(
        #     **data)

        # subprocess.check_call(command, shell=True)
