import os
import pandas as pd
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer


def calculate_network_size(filters, layers):
    size = 0
    image_size = 2**layers
    for _ in range(layers):
        size += filters * image_size
        image_size /= 2
    return size


def calculate_network_flops(filters, layers):
    flops = 0
    image_size = 2**layers
    for _ in range(layers):
        flops += 9 * filters**2 * image_size**2
        image_size /= 2
    return flops


def calculate_network_parameters(filters, layers):
    parameters = 0
    for _ in range(layers):
        parameters += 9 * filters**2
    return parameters


class SearchActivationVisualizer(BaseVisualizer):
    def plot(self, results):

        results = results[results.max_train_size == 1000].copy()
        results = results[results.l1_reg == 0].copy()
        results = results[[
            "model_filters", "model_layers", "model_activation", "val_loss",
            "class"
        ]]
        results = results.rename(
            columns={
                "model_filters": "filters",
                "model_layers": "layers",
                "model_activation": "activation"
            })
        results["size"] = results.apply(
            lambda row: calculate_network_size(row["filters"], row["layers"]),
            axis=1)
        results["flops"] = results.apply(
            lambda row: calculate_network_flops(row["filters"], row["layers"]),
            axis=1)
        results["parameters"] = results.apply(
            lambda row: calculate_network_parameters(row["filters"], row[
                "layers"]),
            axis=1)

        activations = results["activation"].unique().tolist()
        activation_map = dict([(a, activations.index(a)) for a in activations])
        results["activation_key"] = results["activation"].map(activation_map)

        results = results[[
            "size", "flops", "parameters", "activation", "activation_key",
            "class", "val_loss"
        ]]
        results = results.groupby([
            "size", "flops", "parameters", "activation", "activation_key",
            "class"
        ]).min().reset_index()

        results = results.sort_values(by="val_loss", ascending=True)
        # results = results[results["activation"] != "linear"]

        for clazz in results["class"].unique().tolist():
            clazz_results = results[results["class"] == clazz]
            for dimension in ["flops", "parameters", "size"]:
                dimension_results = clazz_results[[
                    dimension, "activation", "val_loss"
                ]].copy()

                min_dimension = dimension_results[dimension].min()

                dimension_results.loc[:, dimension] = dimension_results[
                    dimension].div(min_dimension)

                dimension_results = dimension_results.groupby(
                    [dimension, "activation"]).min().reset_index()
                dimension_results = dimension_results.sort_values(dimension)

                dimension_results = dimension_results.pivot(
                    index=dimension, columns="activation", values="val_loss")
                plot = dimension_results.plot()
                fig = plot.get_figure()
                plt.title("Class %s - Val Loss vs. Network %s" %
                          (clazz, dimension.title()))
                plt.ylabel("Val Loss")
                plt.xlabel("Relative Netowrk %s" % dimension.title())
                outfile = os.path.join(
                    self.data_dir,
                    "class_%s_val_loss_vs_%s.png" % (clazz, dimension.title()))
                fig.savefig(outfile,
                            dpi=150,
                            bbox_inches='tight',
                            pad_inches=0.5)
                plt.close()

    def execute(self):
        csv_file = os.path.join(self.data_dir, "train_results.csv")
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        results = pd.read_csv(csv_file)
        self.plot(results.copy())
