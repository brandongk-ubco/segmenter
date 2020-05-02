import os
import json
import sys
import importlib
from launcher import Task
import pathlib
from segmenter.datasets import Datasets
import glob


class InitializeDataset(Task):
    name = 'initialize-dataset'

    def __init__(self, args):
        self.args = args
        self.dataset_name = args["dataset"]
        self.dataset_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), self.dataset_name)
        self.dataset = Datasets.get(self.dataset_name)(self.dataset_dir)

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(InitializeDataset.name,
                                           help='Initialize a dataset.')
        command_parser.add_argument("dataset",
                                    type=str,
                                    choices=Datasets.choices(),
                                    help='the dataset to initialize.')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([args["dataset"].name])

    def execute(self) -> None:
        self.dataset.initialize()


class VisualizeDataset(Task):

    name = "visualize-dataset"

    def __init__(self, args):
        self.args = args
        self.dataset_name = args["dataset"]
        self.dataset_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), self.dataset_name)
        self.dataset = Datasets.get(self.dataset_name)(self.dataset_dir)
        self.output_dir = os.path.join(os.path.abspath(args["data_dir"]),
                                       self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(VisualizeDataset.name,
                                           help='Visualize a dataset.')
        command_parser.add_argument("dataset",
                                    type=str,
                                    choices=Datasets.choices(),
                                    help='the dataset to visualize.')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([args["dataset"].name])

    def execute(self) -> None:
        from matplotlib import pyplot as plt
        import numpy as np

        classes = self.dataset.get_classes()

        for instance in glob.glob("{}/*.npz".format(self.output_dir)):
            name = os.path.basename(instance)[:-4]
            print(name)
            instance_data = np.load(instance)
            image = instance_data["image"]
            mask = instance_data["mask"]

            fig = plt.figure(constrained_layout=True)

            gs = fig.add_gridspec(2, mask.shape[-1])
            fig_image = fig.add_subplot(gs[:, 0])
            fig_image.axis('off')
            fig_image.set_title('Image')
            fig_image.imshow(image, cmap='gray')

            for mask_idx in range(mask.shape[-1]):
                fig_mask = fig.add_subplot(gs[mask_idx, 1])
                fig_mask.axis('off')
                fig_mask.set_title(classes[mask_idx])
                fig_mask.imshow(mask[:, :, mask_idx], cmap='gray')

            out_file = os.path.join(os.path.dirname(instance),
                                    "{}.png".format(name))
            plt.savefig(out_file, dpi=100, bbox_inches='tight', pad_inches=0.5)

            plt.close()


class ProcessDataset(Task):

    name = 'process-dataset'

    def __init__(self, args):
        self.args = args
        self.dataset_name = args["dataset"]
        self.dataset_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), self.dataset_name)
        self.dataset = Datasets.get(self.dataset_name)(self.dataset_dir)
        self.output_dir = os.path.join(os.path.abspath(args["data_dir"]),
                                       self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def arguments(parser) -> None:
        command_parser = parser.add_parser(ProcessDataset.name,
                                           help='Process a dataset.')
        command_parser.add_argument("dataset",
                                    type=str,
                                    choices=Datasets.choices(),
                                    help='the dataset to process.')

    @staticmethod
    def arguments_to_cli(args) -> str:
        return " ".join([args["dataset"].name])

    def execute(self) -> None:
        import numpy as np

        os.makedirs(self.output_dir, exist_ok=True)

        class_members = self.dataset.get_class_members()
        classes = self.dataset.get_classes()

        for c in classes:
            assert c in class_members.keys(
            ), "Class %s not defined in class members" % c
            assert isinstance(
                c,
                str), "Class %s should be a string, but found %s" % (c,
                                                                     type(c))

        for c in class_members.keys():
            assert c in classes, "Class %s not defined in classes" % c

        class_counts = [
            len(class_members[c]["eval_instances"]) +
            len(class_members[c]["train_instances"])
            for c in class_members.keys()
        ]
        print("Class counts: %s" % dict(zip(classes, class_counts)))

        for num_folds in range(2, 21):
            folds = self.dataset.split_folds(class_members, num_folds)
            with open(
                    os.path.join(self.output_dir, "%s-folds.json" % num_folds),
                    "w") as outfile:
                json.dump({"folds": folds}, outfile, indent=4)

        mean, std = self.dataset.process_images(
            lambda image, mask, name: np.savez_compressed(
                os.path.join(self.output_dir, name), image=image, mask=mask))

        print("Mean = {}".format(mean))
        print("Std = {}".format(std))

        with open(os.path.join(self.output_dir, "classes.json"),
                  "w") as outfile:
            json.dump(
                {
                    "properties": {
                        "mean": mean,
                        "std": std
                    },
                    "classes": class_members,
                    "class_order": [str(s) for s in classes]
                },
                outfile,
                indent=4)


tasks = [InitializeDataset, ProcessDataset, VisualizeDataset]
