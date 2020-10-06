import json
from segmenter.helpers.p_tqdm import t_map as mapper
import glob
import numpy as np
import os
from skimage import measure
from matplotlib import pyplot as plt
from itertools import chain


class DatasetStatistics:
    coverage_results = {"instances": []}
    component_results = {"instances": []}

    def __init__(self, dataset, src_dir):
        self.dataset = dataset
        self.src_dir = src_dir
        self.classes = self.dataset.get_classes()
        with open(os.path.join(src_dir, "classes.json"), "r") as infile:
            class_members = json.load(infile)
        self.class_members = dict([
            (k, v["eval_instances"] + v["train_instances"])
            for k, v in class_members["classes"].items()
        ])
        for clazz in self.classes:
            self.coverage_results[clazz] = []
            self.component_results[clazz] = []

    def execute_coverage(self, result_path):
        result = np.load(result_path)
        name = os.path.basename(result_path)[:-4]
        mask = result["mask"]
        self.coverage_results["instances"].append(name)
        for i, clazz in enumerate(self.classes):
            if name not in self.class_members[clazz]:
                continue
            class_mask = mask[:, :, i]
            coverage = np.sum(class_mask) / class_mask.size
            self.coverage_results[clazz].append(coverage)

    def execute_components(self, result_path):
        result = np.load(result_path)
        name = os.path.basename(result_path)[:-4]
        mask = result["mask"]
        self.component_results["instances"].append(name)
        for i, clazz in enumerate(self.classes):
            if name not in self.class_members[clazz]:
                continue
            class_mask = mask[:, :, i]
            components = measure.label(class_mask, background=0)
            component_sizes = []
            for component in range(1, np.max(components) + 1):
                size = int(np.sum(components == component))
                component_sizes.append(size / class_mask.size)
            if len(component_sizes) == 0:
                raise ValueError("No components for {} Class {}".format(
                    name, clazz))
            self.component_results[clazz].append(component_sizes)

    def calculate_coverage(self):
        coverage_file = os.path.join(self.src_dir, "coverage.json")
        if os.path.exists(coverage_file):
            with open(coverage_file, "r") as outfile:
                self.coverage_results = json.load(outfile)
        else:
            print("Calculating Coverage")
            mapper(self.execute_coverage, self.collect_results())

            with open(coverage_file, "w") as outfile:
                json.dump(self.coverage_results, outfile)

    def calculate_components(self):
        component_file = os.path.join(self.src_dir, "components.json")
        if os.path.exists(component_file):
            with open(component_file, "r") as outfile:
                self.component_results = json.load(outfile)
        else:
            print("Calculating Components")
            mapper(self.execute_components, self.collect_results())

            with open(component_file, "w") as outfile:
                json.dump(self.component_results, outfile)

    def plot_coverage(self):
        for i, clazz in enumerate(self.classes):
            coverage_file = os.path.join(self.src_dir,
                                         "{}_coverage.png".format(clazz))
            coverages = [
                100 * j for j in self.coverage_results[self.classes[i]]
            ]

            fig = plt.figure()

            fig.set_size_inches(10, 5)

            bins = np.unique(
                np.round(np.linspace(0, max(coverages), num=100), 1))

            plt.hist(coverages,
                     bins=bins,
                     weights=100 * np.ones(len(coverages)) / len(coverages))

            plt.ylabel("Frequency (%)")

            plt.xlabel(
                "Coverage (%): Mean {:1.2f}, Median {:1.2f}, St. Dev. {:1.2f}".
                format(np.mean(coverages), np.median(coverages),
                       np.std(coverages)))

            plt.xticks(bins[::3])
            plt.tick_params(axis='x', rotation=90)

            title = "Coverage Distribution for Class {}".format(clazz)

            plt.title(title)

            fig.savefig(coverage_file,
                        dpi=150,
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()

    def plot_component_size(self):
        for i, clazz in enumerate(self.classes):
            component_file = os.path.join(
                self.src_dir, "{}_component_size.png".format(clazz))

            sizes = [
                100 * j for j in list(
                    chain.from_iterable(self.component_results[
                        self.classes[i]]))
            ]

            fig = plt.figure()

            bins = np.unique(np.round(np.linspace(0, max(sizes), num=100), 1))

            plt.hist(sizes,
                     bins=bins,
                     weights=100 * np.ones(len(sizes)) / len(sizes))

            plt.ylabel("Frequency (%)")

            plt.xlabel(
                "Component Size (%): Mean {:1.2f}, Median {:1.2f}, St. Dev. {:1.2f}"
                .format(np.mean(sizes), np.median(sizes), np.std(sizes)))

            plt.xticks(bins[::5])
            plt.tick_params(axis='x', rotation=90)

            title = "Component Size Distribution for Class {}".format(clazz)

            plt.title(title)

            fig.savefig(component_file,
                        dpi=150,
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()

    def plot_component_count(self):
        for i, clazz in enumerate(self.classes):
            component_file = os.path.join(
                self.src_dir, "{}_component_count.png".format(clazz))

            counts = [len(j) for j in self.component_results[self.classes[i]]]

            fig = plt.figure()
            bins = np.arange(1, max(counts))

            plt.hist(counts,
                     bins=bins,
                     weights=100 * np.ones(len(counts)) / len(counts),
                     align='left')

            plt.ylabel("Frequency (%)")

            plt.xticks(bins[:-1])

            plt.xlabel("Components per Image")

            plt.tick_params(axis='x', rotation=90)

            title = "Components per Image Distribution for Class {}".format(
                clazz)

            plt.title(title)

            fig.savefig(component_file,
                        dpi=150,
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()

    def plot_number_of_instances(self):

        instance_file = os.path.join(self.src_dir, "instance_count.png")
        instance_counts = dict([(k, len(v))
                                for k, v in self.class_members.items()])
        classes = np.arange(len(instance_counts.keys()))
        counts = [v for k, v in instance_counts.items()]

        fig, ax = plt.subplots()
        rects = plt.bar(classes, counts, align='center', alpha=0.5)
        plt.xticks(classes, instance_counts.keys())
        plt.ylabel('Number of Instances')
        plt.title('Count of Instances by Class')

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                '{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -12),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom')

        fig.savefig(instance_file,
                    dpi=150,
                    bbox_inches='tight',
                    pad_inches=0.5)
        plt.close()

    def execute(self):
        self.calculate_coverage()
        self.calculate_components()
        self.plot_coverage()
        self.plot_component_size()
        self.plot_component_count()
        self.plot_number_of_instances()

    def collect_results(self):
        return sorted(glob.glob("{}/*.npz".format(self.src_dir)))
