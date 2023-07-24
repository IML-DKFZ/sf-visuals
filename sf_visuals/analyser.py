import itertools
import os
from functools import cache, cached_property
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from fd_shifts.analysis import PlattScaling, confid_scores
from loguru import logger
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sf_visuals.utils.utils import (
    getdffromarrays,
    kmeans_cluster_representative_without_failurelabel,
)


class Analyser:
    """Build a class holding experimental data to visualy analyse.

    Attributes:
        accuracies_dict:
        n_classes:
        color_map:
        colors:
    """

    def __init__(
        self,
        path: Path | str,
        base_path: Path | str,
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self.__path = path / "test_results"
        logger.info("Loading Data for {}", path)
        self.__base_path = base_path
        self.__raw_output = np.load(self.__path / "raw_output.npz")["arr_0"]
        self.__raw_output_dist = np.load(self.__path / "raw_output_dist.npz")["arr_0"]
        self.__external_confid = np.load(self.__path / "external_confids.npz")["arr_0"]
        self.__softmax_output = self.__raw_output[:, :-2]
        self.__out_class = np.argmax(np.squeeze(self.__softmax_output), axis=1)
        self.__labels = np.squeeze(np.asarray(self.__raw_output[:, -2])).astype(int)
        self.__encoded_output = np.load(self.__path / "encoded_output.npz")["arr_0"]

        classes = np.unique(self.__labels)
        self.__class2name = dict(zip(classes, classes))
        self.__class2plot = tuple(classes)

        with (path / "hydra" / "config.yaml").open() as file:
            config = yaml.safe_load(file)

        flat_test_set_list = []
        if config["eval"]["val_tuning"]:
            flat_test_set_list.append("validation")
        for _, datasets in config["eval"]["query_studies"].items():
            if isinstance(datasets, list):
                flat_test_set_list.extend(list(datasets))
            else:
                flat_test_set_list.append(datasets)

        def _rename_datasets(dataset: str):
            return dataset.replace("${data.dataset}", "iid").replace(
                config["data"]["dataset"], ""
            )

        self.__confid_names = config["eval"]["confidence_measures"]["test"]

        flat_test_set_list = list(map(_rename_datasets, flat_test_set_list))

        self.__ls_testsets = flat_test_set_list
        self.__test_datasets = flat_test_set_list[1:]

        self.__test_datasets_length = len(self.__ls_testsets)
        self.__csvs = []
        self.accuracies_dict = {}
        for i in range(int(self.__test_datasets_length)):
            try:
                attributions = pd.read_csv(self.__path / f"attributions{i}.csv")
            except:
                attributions = pd.read_csv(self.__path / "attributions.csv")

            attributions.filepath = attributions.filepath.str.replace(
                "/dkfz/cluster/gpu/data/OE0612/l049e/", ""
            )
            self.__csvs.append(attributions)
        self.__softmax_beginning = np.squeeze(self.__softmax_output)
        self.n_classes = len(np.unique(self.__labels))

        self.color_map = {
            "TN": "rgb(26,150,65)",
            "FN": "rgb(166,217,106)",
            "FP": "rgb(253,174,97)",
            "TP": "rgb(215,25,28)",
        }
        self.colors = [
            "rgb(31, 119, 180)",
            "rgb(255, 127, 14)",
            "rgb(44, 160, 44)",
            "rgb(214, 39, 40)",
            "rgb(148, 103, 189)",
            "rgb(140, 86, 75)",
            "rgb(227, 119, 194)",
            "rgb(127, 127, 127)",
            "rgb(188, 159, 34)",
            "rgb(23, 190, 207)",
            "rgb(41, 119, 180)",
            "rgb(255, 117, 14)",
            "rgb(54, 160, 44)",
            "rgb(214, 29, 40)",
            "rgb(158, 103, 189)",
            "rgb(140, 76, 75)",
            "rgb(237, 119, 194)",
            "rgb(127, 117, 127)",
            "rgb(198, 189, 34)",
            "rgb(23, 180, 207)",
        ]
        logger.info("Done initializating Analyzer")

    @property
    def confid_names(self):
        return self.__confid_names

    @property
    def classes(self):
        return list(self.__class2name.values())

    @classes.setter
    def classes(self, values: dict):
        self.__class2name |= values

    @property
    def testsets(self):
        return self.__test_datasets

    @cache
    def get_mcd_outputs(self, testset: str):
        i = self.__ls_testsets.index(testset)
        boolarray = self.__encoded_output[:, -1] == i
        return self.__raw_output_dist[boolarray]

    @cache
    def embedding(self, testset: str):
        """Compute the embedding for a given testset.

        Args:
            testset: The testset to compute the embedding for.

        Returns:
            The embedding for the given testset.
        """
        logger.info("Computing embedding for {}", testset)

        i = self.__ls_testsets.index(testset)
        boolarray = self.__encoded_output[:, -1] == i
        predicted = self.__out_class[boolarray]
        encoded = self.__encoded_output[boolarray][:, :-1]
        softmax = self.__softmax_beginning[boolarray]
        ext = self.__external_confid[boolarray]
        y_true = self.__labels[boolarray]

        folder2create = (
            Path.cwd() / "outputs" / self.__path.relative_to(self.__base_path).parent
        )
        logger.warning(f"{folder2create=}")
        check_file = folder2create / testset / "dataframe.csv"
        logger.warning(f"{check_file=}")

        if check_file.is_file():
            logger.info("Found cached embedding {}", check_file)
            df = pd.read_csv(check_file, index_col=False)
        else:
            pca50 = PCA(n_components=50)
            pca_encoded50 = pca50.fit_transform(encoded)
            tsne_encoded3_set = TSNE(
                n_components=3, init="pca", learning_rate=200
            ).fit_transform(pca_encoded50)
            testset_csv = self.__csvs[i]
            fp = testset_csv.filepath
            df = getdffromarrays(
                labels=y_true,
                out_class=predicted,
                tsne_encoded3=tsne_encoded3_set,
                softmax=softmax,
                filepath=fp,
            )

            logger.info("Writing embedding to cache {}", check_file)
            check_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(check_file, index=False)

        df["testset"] = testset
        df["ext_confid"] = ext

        return df

    @cache
    def get_platt_scaling(self, csf: str):
        df = self.embedding("validation")

        if "det" in csf:
            confid_func = confid_scores.get_confid_function(csf)
            df["_confid"] = confid_func(df.filter(like="softmax").to_numpy())
            logger.debug(f"{df['_confid']=}")
        elif "mcd" in csf:
            confid_func = confid_scores.get_confid_function(csf)
            mcd_dist = self.get_mcd_outputs("validation")
            df["_confid"] = confid_func(mcd_dist.mean(axis=2), mcd_dist)
        elif "ext" in csf:
            df["_confid"] = df["ext_confid"]
        else:
            raise NotImplementedError

        return PlattScaling(
            df["_confid"].to_numpy(),
            (df["predicted"] == df["label"]).astype(int).to_numpy(),
        )

    @cache
    def plot_latentspace(
        self,
        testsets: tuple[str, ...],
        classes2plot: tuple[int] | None = None,
        coloring: Literal[
            "confidence", "source-target", "class-confusion"
        ] = "confidence",
        csf: str = "det_mcp",
    ):
        df = pd.concat([self.with_confid(t, csf) for t in testsets])

        if classes2plot is None:
            classes2plot = self.__class2plot

        vmin = df["confid"].min()
        vmax = df["confid"].max()

        xmin = np.percentile(df["0"], 1)
        xmax = np.percentile(df["0"], 99)
        ymin = np.percentile(df["1"], 1)
        ymax = np.percentile(df["1"], 99)
        zmin = np.percentile(df["2"], 1)
        zmax = np.percentile(df["2"], 99)

        markers = [
            "circle",
            "cross",
            "diamond",
            "square",
            "circle-open",
            "diamond-open",
            "square-open",
        ]
        colors = [["cyan", "blue"], ["gray", "magenta"]]
        colorscales = [
            [(0, "#e46c00"), (1, "#67da40")],
            [(0, "#56c1ff"), (1, "#ee230c")],
        ]

        def filter_by(
            data: pd.DataFrame,
            by: Literal["correct", "class", "testset"],
            value: bool | int | str,
        ):
            match by:
                case "correct":
                    return data[(data.label == data.predicted) == value]
                case "class":
                    return data[data.label == value]
                case "testset":
                    return (
                        data[data.testset == "iid"]
                        if value == "iid"
                        else data[data.testset != "iid"]
                    )

        if coloring == "confidence":
            traces = [
                (
                    [("correct", correct), ("class", c)],
                    f"Class {c}",
                    lambda data, crl=correct, cl=c: {
                        "size": 5,
                        "cmin": vmin,
                        "cmax": vmax,
                        "color": data["confid"],
                        "colorscale": colorscales[0 if crl else 1],
                        "colorbar": {
                            "title": {
                                "text": f"Confidence (classification {'success' if crl else 'failure'})",
                                "side": "top",
                            },
                            "orientation": "h",
                            "y": 0.99 - 0.10 * (1 if crl else 0),
                            "thickness": 0.02,
                            "thicknessmode": "fraction",
                        }
                        if cl == 0
                        else None,
                        "symbol": markers[cl % len(markers)],
                    },
                    correct,
                )
                for correct, c in itertools.product([True, False], classes2plot)
            ]
        elif coloring == "source-target":
            colors = [["cyan", "blue"], ["gray", "magenta"]]
            traces = [
                (
                    [("testset", t), ("class", c)],
                    f"C={c}, T={t}",
                    lambda data, tl=t, cl=c: {
                        "size": 5,
                        "cmin": vmin,
                        "cmax": vmax,
                        "color": colors[0 if tl == "iid" else 1][
                            cl % len(colors[0 if tl == "iid" else 1])
                        ],
                        "symbol": markers[cl % len(markers)],
                    },
                    True,
                )
                for t, c in itertools.product(df.testset.unique(), classes2plot)
            ]
        elif coloring == "class-confusion":
            colors = [["#00876c", "#89bf77"], ["#d43d51", "#f59b56"]]
            traces = [
                (
                    [("correct", correct), ("class", c)],
                    f"{'C' if correct else'Inc'}orrect, C={c}",
                    lambda data, crl=correct, cl=c: {
                        "size": 5,
                        "cmin": vmin,
                        "cmax": vmax,
                        "color": colors[0 if crl else 1][
                            cl % len(colors[0 if crl else 1])
                        ],
                        "symbol": markers[cl % len(markers)],
                    },
                    True,
                )
                for correct, c in itertools.product([True, False], classes2plot)
            ]
        else:
            raise ValueError

        fig = go.Figure()

        for filters, label, markers_fn, show_legend in traces:
            data = df
            for by, value in filters:
                data = filter_by(data, by, value)
            fig.add_trace(
                go.Scatter3d(
                    x=data["0"],
                    y=data["1"],
                    z=data["2"],
                    opacity=0.4,
                    mode="markers",
                    name=label,
                    customdata=data.label.astype(str)
                    .str.cat(data.predicted.astype(str), sep=",")
                    .str.cat(data["confid"].astype(str), sep=","),
                    text=data["filepath"],
                    hoverinfo="name",
                    marker=markers_fn(data),
                    showlegend=show_legend,
                )
            )

        fig.update_layout(
            coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside")
        )
        fig.update_layout(legend={"itemsizing": "constant", "orientation": "h"})
        return fig

    @cache
    def representative(self, testset: str, cls: int, csf: str = "det_mcp"):
        df = self.with_confid(testset, csf)
        data = kmeans_cluster_representative_without_failurelabel(
            dataframe=df,
            cla=cls,
        )
        return data[
            ["filepath", "predicted", "label", "confid", "0", "1", "2"]
        ].to_dict("records")

    @cache
    def with_confid(self, testset, csf):
        df = self.embedding(testset)
        if "det" in csf:
            confid_func = confid_scores.get_confid_function(csf)
            df["confid"] = confid_func(df.filter(like="softmax").to_numpy())
        elif "mcd" in csf:
            confid_func = confid_scores.get_confid_function(csf)
            mcd_dist = self.get_mcd_outputs(testset)
            df["confid"] = confid_func(mcd_dist.mean(axis=2), mcd_dist)
        elif "ext" in csf:
            df["confid"] = df["ext_confid"]
        else:
            raise NotImplementedError

        if any(
            cfd in csf for cfd in ["_pe", "_ee", "_mi", "_sv", "bpd", "maha", "_mls"]
        ):
            df["confid"] = self.get_platt_scaling(csf)(df["confid"].to_numpy())

        return df

    @cache
    def overconfident(self, testset: str, csf: str = "det_mcp"):
        """
        return matplotlib plot with overconfident images
        """
        df = self.with_confid(testset, csf)
        df = df[df.label != df.predicted]
        df = df.sort_values("confid", ascending=False, ignore_index=True)
        return (
            df[["filepath", "confid", "predicted", "label"]]
            .drop_duplicates(["filepath", "predicted", "label"])
            .iloc[:3, :]
            .to_dict("records")
        )

    def get_coords_from_filename(self, filename: str, testset: str):
        df = self.embedding(testset)
        return df[df.filepath == filename].to_dict("records")[0]
