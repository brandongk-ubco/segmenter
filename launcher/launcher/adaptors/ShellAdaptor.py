from launcher.adaptors.Adaptor import Adaptor
import os
import subprocess


class ShellAdaptor(Adaptor):

    @staticmethod
    def arguments(parser):
        group = parser.add_argument_group(
            'shell', 'Arguments for the Shell Adaptor')
        group.add_argument("--shell", type=bool,
                           default=True, help='run as a shell.')

    @staticmethod
    def execute(task, args) -> None:
        task(args).execute()
