from launcher.adaptors.Adaptor import Adaptor
import os
import subprocess
import sys
import argparse


class DockerAdaptor(Adaptor):
    @staticmethod
    def arguments(parser):
        group = parser.add_argument_group('docker',
                                          'Arguments for the Docker Adaptor')
        group.add_argument(
            "--image",
            type=str,
            default=os.path.basename(os.path.abspath(".")),
            help=
            'image to run as.  Defaults to the directory name in which the command is run.'
        )
        group.add_argument("--gpus",
                           type=bool,
                           default=True,
                           help='Enable gpus in docker container.')

    @staticmethod
    def execute(task, args) -> None:
        subprocess.check_call("docker build . -t {}".format(args["image"]),
                              shell=True)

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
            "gpus": "--gpus all" if args["gpus"] else "",
        }

        if "JOB_CONFIG" in os.environ:
            data["job_config"] = os.environ["JOB_CONFIG"]

        if "job_config" in args:
            data["job_config"] = args["job_config"]

        command = "docker run -v {path}/{project}:/src/{project} -v {data_dir}:/data -v {output_dir}:/output -u {uid}:{gid} {gpus} --entrypoint {command} -e JOB_CONFIG={job_config} -it {image} {args}".format(
            **data)
        print(command)
        subprocess.check_call(command, shell=True)
