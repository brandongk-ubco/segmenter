import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


class SearchParallelCoordinatesVisualizer(BaseVisualizer):
    def plot(self, results):
        results = results[[
            "model_filters", "model_layers", "model_activation", "l1_reg",
            "val_loss"
        ]]
        results = results.rename(
            columns={
                "model_filters": "filters",
                "model_layers": "layers",
                "model_activation": "activation"
            })
        activations = results["activation"].unique().tolist()
        print(activations)
        activation_map = dict([(a, activations.index(a)) for a in activations])
        results["activation_key"] = results["activation"].map(activation_map)
        results = results.groupby(
            ['filters', 'layers', "activation", "activation_key",
             "l1_reg"]).min().reset_index()

        results = results.sort_values(by="val_loss", ascending=True)
        slices = 20
        num_per_slice = len(results) // slices
        for slce in range(0, slices):
            start = slce * num_per_slice
            end = (slce + 1) * num_per_slice
            print(start, end)
            result_slice = results[start:end]

            fig = go.Figure(data=go.Parcoords(
                line=dict(color=result_slice['val_loss'],
                          colorscale='Electric_r',
                          showscale=True),
                dimensions=list([
                    dict(label='filters',
                         values=result_slice["filters"],
                         tickvals=results["filters"].unique().tolist(),
                         range=[
                             min(results["filters"]),
                             max(results["filters"])
                         ]),
                    dict(
                        label='layers',
                        values=result_slice["layers"],
                        tickvals=results["layers"].unique().tolist(),
                        range=[min(results["layers"]),
                               max(results["layers"])]),
                    dict(label='activation',
                         values=result_slice["activation_key"],
                         tickvals=list(activation_map.values()),
                         ticktext=list(activation_map.keys()),
                         range=[
                             min(results["activation_key"]),
                             max(results["activation_key"])
                         ]),
                    dict(
                        label='l1_reg',
                        values=result_slice["l1_reg"],
                        tickvals=results["l1_reg"].unique().tolist(),
                        range=[min(results["l1_reg"]),
                               max(results["l1_reg"])],
                        ticktext=[
                            "{:.1E}".format(t)
                            for t in results["l1_reg"].unique().tolist()
                        ]),
                    dict(label='val_loss',
                         values=result_slice["val_loss"],
                         range=[
                             max(result_slice["val_loss"]),
                             min(result_slice["val_loss"])
                         ]),
                ])))
            fig.update_layout(
                title=
                "Validation Loss Percentiles %d%% to %d%% Grid Search Results"
                % (100 - (slce + 1) / slices * 100, 100 - slce / slices * 100))
            outfile = os.path.join(
                self.data_dir,
                "parallel_coordinates_%d.png" % (100 - slce / slices * 100))
            fig.write_image(outfile)
        # plot = pd.plotting.parallel_coordinates(results, "quantile")
        # fig = plot.get_figure()
        # plot.legend('')
        #
        # fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
        # plt.close()

    def execute(self):
        csv_file = os.path.join(self.data_dir, "train_results.csv")
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        results = pd.read_csv(csv_file)
        self.plot(results.copy())
