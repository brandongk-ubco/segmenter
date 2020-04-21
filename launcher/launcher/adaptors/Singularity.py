from launcher.adaptors.Adaptor import Adaptor
import os
import subprocess


class SingularityAdaptor(Adaptor):
    @staticmethod
    def arguments(parser):
        parser.add_argument_group('singularity',
                                  'Arguments for the Singularity Adaptor')

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
            "gpus": "--nv" if args["gpus"] else ""
        }

        if "job_config" in args:
            os.environ["JOB_CONFIG"] = args["job_config"]

        command = "singularity exec -c -B {path}/{project}:/src/{project} -B {data_dir}:/data -B {output_dir}:/output {gpus} --pwd /src image.sif {command} {args}".format(
            **data)
        print(command)
        subprocess.check_call(command, shell=True)

        # command = "docker run -v {path}/{project}:/src/{project} -v {data_dir}:/data -v {output_dir}:/output -u {uid}:{gid} {gpus} --entrypoint {command} -it {image} {args}".format(
        #     **data)

        # subprocess.check_call(command, shell=True)
