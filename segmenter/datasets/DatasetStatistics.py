import json
from segmenter.helpers.p_tqdm import t_map as mapper
import glob
import numpy as np
import os
from skimage import measure
from matplotlib import pyplot as plt


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
            for component in range(1, np.max(components)):
                size = int(np.sum(components == component))
                component_sizes.append(size)
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

            fig, (ax1,
                  ax2) = plt.subplots(1,
                                      2,
                                      gridspec_kw={'width_ratios': [3, 1]})
            fig.tight_layout()

            fig.set_size_inches(10, 5)

            bins = np.unique(
                np.round(np.linspace(0, max(coverages), num=100), 1))

            ax1.hist(coverages,
                     bins=bins,
                     weights=100 * np.ones(len(coverages)) / len(coverages))

            ax1.set(ylabel="Frequency (%)")

            ax1.set(
                xlabel=
                "Coverage (%): Mean {:1.2f}, Median {:1.2f}, St. Dev. {:1.2f}".
                format(np.mean(coverages), np.median(coverages),
                       np.std(coverages)))

            ax1.set_xticks(bins[::3])
            ax1.tick_params(axis='x', rotation=90)

            plt.subplots_adjust(hspace=.2)

            ax2.boxplot(coverages, vert=True)
            ax2.set(ylabel="Coverage (%)")
            ax2.set_xticks([])

            title = "Coverage Distribution for Class {}".format(clazz)

            fig.suptitle(title, y=1.05, fontsize=14)

            fig.savefig(coverage_file,
                        dpi=70,
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()

    def plot_component_size(self):
        for i, clazz in enumerate(self.classes):
            component_file = os.path.join(
                self.src_dir, "{}_component_size.png".format(clazz))
            import pdb
            pdb.set_trace()
            sizes = [100 * j for j in self.component_results[self.classes[i]]]

            fig, (ax1,
                  ax2) = plt.subplots(1,
                                      2,
                                      gridspec_kw={'width_ratios': [3, 1]})
            fig.tight_layout()

            fig.set_size_inches(10, 5)

            bins = np.unique(
                np.round(np.linspace(0, max(coverages), num=100), 1))

            percentages, _bins, _patches = ax1.hist(
                coverages,
                bins=bins,
                weights=100 * np.ones(len(coverages)) / len(coverages))

            percentile_to_find = 99
            percentile = np.percentile(coverages, percentile_to_find)

            ax1.set(ylabel="Frequency (%): Peak {:1.2f}% at {:1.2f}.".format(
                np.max(percentages), bins[np.argmax(percentages)]))

            ax1.set(ylim=[0, percentile])

            ax1.set(
                xlabel=
                "Coverage (%): Mean {:1.2f}, Median {:1.2f}, St. Dev. {:1.2f}".
                format(np.mean(coverages), np.median(coverages),
                       np.std(coverages)))

            ax1.set_xticks(bins[::3])
            ax1.tick_params(axis='x', rotation=90)

            plt.subplots_adjust(hspace=.2)

            ax2.boxplot(coverages, vert=True)
            ax2.set(ylabel="Coverage (%)")
            ax2.set_xticks([])

            title = "Coverage Distribution for Class {}".format(clazz)

            fig.suptitle(title, y=1.05, fontsize=14)

            fig.savefig(coverage_file,
                        dpi=70,
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()

    def execute(self):
        self.calculate_coverage()
        self.calculate_components()
        # self.plot_coverage()
        self.plot_component_size()

    def collect_results(self):
        return glob.glob("{}/*.npz".format(self.src_dir))
