from io import BytesIO
import itertools
import os
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sf_visuals.utils.utils import (
    getdffromarrays, kmeans_cluster_representative_without_failurelabel,
    overconfident_images, underconfident_images)


class Analyser:
    """
    Build a class holding experimental data to visualy analyse
        path:  "../experiemnt_group_name/experiment_name"
        class2plot:   dict({0:"myclassname0",...}) conatianing a mapping of integer classes to real names
        ls_testsets:   ["nameoftestset",...] a list with names of all testsets
        class2plot and test_datasets are two lists with a subset of classes/ testset names for which to generate outputs. (output can be quite large)
    """

    def __init__(
        self,
        path: Path | str,
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self.__path = path / "test_results"
        # Loading Data from init parameters
        self.__raw_output = np.load(self.__path / "raw_output.npz")["arr_0"]
        self.__softmax_output = self.__raw_output[:, :-2]
        self.__out_class = np.argmax(np.squeeze(self.__softmax_output), axis=1)
        self.__labels = np.squeeze(np.asarray(self.__raw_output[:, -2])).astype(int)
        self.__encoded_output = np.load(self.__path / "encoded_output.npz")["arr_0"]
        self.__dataset_idx = np.squeeze(np.asarray(self.__raw_output[:, -1])).astype(
            int
        )

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

        flat_test_set_list = list(map(_rename_datasets, flat_test_set_list))

        self.__ls_testsets = flat_test_set_list
        self.__test_datasets = flat_test_set_list[1:3]

        self.__test_datasets_length = len(self.__ls_testsets)
        self.__csvs = []
        self.__dataframes = {}
        self.__encoderls = {}
        self.accuracies_dict = {}
        for i in range(int(self.__test_datasets_length)):
            try:
                attributions = pd.read_csv(self.__path / f"attributions{i}.csv")
            except:
                attributions = pd.read_csv(self.__path / "attributions.csv")

            attributions.filepath = attributions.filepath.str.replace(
                "/dkfz/cluster/gpu/data/OE0612/l049e", "/home/t974t/Data/levin"
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
    def embedding(self, testset: str):
        i = self.__ls_testsets.index(testset)
        study = str(self.__ls_testsets[i])
        boolarray = self.__encoded_output[:, -1] == i
        len_testset = np.sum(boolarray)
        predicted = self.__out_class[boolarray]
        baseline_class = mode(self.__labels)[0][0]
        encoded = self.__encoded_output[boolarray][:, :-1]
        softmax = self.__softmax_beginning[boolarray]
        y_score = self.__softmax_beginning[boolarray, 1]
        y_true = self.__labels[boolarray]
        subfolders = str(self.__path).split("/")
        cwd = os.getcwd()
        dir = subfolders[-4:-2]
        folder2create = os.path.join(cwd, "outputs", dir[0], dir[1])
        check_file = folder2create + "/" + study + "/dataframe.csv"
        if os.path.exists(check_file):
            df = pd.read_csv(check_file, index_col=False)
        else:
            pca50 = PCA(n_components=50)
            pca_encoded50 = pca50.fit_transform(encoded)
            tsne_encoded3_set = TSNE(
                n_components=3, init="pca", learning_rate=200
            ).fit_transform(pca_encoded50)
            ### create dataframe for each dataset that stores all relevant information
            testset_csv = self.__csvs[i]
            fp = testset_csv.filepath
            ### savety_ check that y_true from raw output == targets from the csv to ensure correct ordering/dataset
            df = getdffromarrays(
                labels=y_true,
                out_class=predicted,
                tsne_encoded3=tsne_encoded3_set,
                softmax=softmax,
                filepath=fp,
            )

        df["testset"] = testset

        testfodlers2create = os.path.join(folder2create, str(testset))
        print(testfodlers2create)
        if not os.path.exists(check_file):
            os.makedirs(testfodlers2create, exist_ok=True)
            df.to_csv(testfodlers2create + "/dataframe.csv", index=False)

        return df

    @cache
    def plot_latentspace(
        self,
        testset: str,
        classes2plot: tuple[int] | None = None,
        coloring: Literal["confidence", "source-target"] = "confidence",
    ):
        if testset == "ALL":
            df = pd.concat([self.embedding(t) for t in self.__test_datasets])
        else:
            df = self.embedding(testset)

        if classes2plot is None:
            classes2plot = self.__class2plot

        df["confid"] = (df.filter(like="softmax").max(axis=1) > 0.8).astype(float)
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
        colorscales = [[(0, "#e46c00"), (1, "#67da40")], [(0, "#56c1ff"), (1, "#ee230c")]]

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
                    f"{'C' if correct else'Inc'}orrect, C={c}",
                    lambda data, crl=correct, cl=c: dict(
                        size=5,
                        cmin=vmin,
                        cmax=vmax,
                        color=data["confid"],
                        colorscale=colorscales[0 if crl else 1],
                        symbol=markers[cl % len(markers)],
                    ),
                )
                for correct, c in itertools.product([True, False], classes2plot)
            ]
        elif coloring == "source-target":
            traces = [
                (
                    [("testset", t), ("class", c)],
                    f"C={c}, T={t}",
                    lambda data, tl=t, cl=c: dict(
                        size=5,
                        cmin=vmin,
                        cmax=vmax,
                        color=colors[0 if tl == "iid" else 1][cl % len(colors[0 if tl == "iid" else 1])],
                        symbol=markers[cl % len(markers)],
                    ),
                )
                for t, c in itertools.product(df.testset.unique(), classes2plot)
            ]
        else:
            raise ValueError

        fig = go.Figure()

        for filters, label, markers_fn in traces:
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
                    text=data["filepath"],
                    marker=markers_fn(data),
                )
            )

        fig.update_layout(
            coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside")
        )
        fig.update_layout(width=1000, height=1000, template="simple_white")
        return fig

    def representative(self, testset, cls):
        df = self.embedding(testset)
        fig = kmeans_cluster_representative_without_failurelabel(
            dataframe=df,
            cla=cls,
        )
        strio = BytesIO()
        fig.savefig(strio, format="png")
        return strio.getvalue()

    def show_underconfident(self):
        """
        return matplotlib plot with underconfident images
        """
        self.__underconfimages = {}
        for testset in self.__test_datasets:
            df = self.__dataframes[testset]
            self.__underconfimages[testset] = underconfident_images(
                df=df, class2name=self.__class2name
            )
        return self.__underconfimages

    def show_overconfident(self):
        """
        return matplotlib plot with overconfident images
        """
        self.__overconfimages = {}
        for testset in self.__test_datasets:
            df = self.__dataframes[testset]
            self.__overconfimages[testset] = overconfident_images(
                df=df, class2name=self.__class2name
            )
        return self.__overconfimages

    def show_representative(self):
        """
        return matplotlib plot with representative images
        """
        self.__representative = {}
        for testset in self.__test_datasets:
            df = self.__dataframes[testset]
            cla_accuracies = self.accuracies_dict[testset]
            representative_per_class = {}
            for cla in self.__class2plot:
                representative_per_class[
                    cla
                ] = kmeans_cluster_representative_without_failurelabel(
                    dataframe=df,
                    cla_accuracies=cla_accuracies,
                    cla=cla,
                    class2name=self.__class2name,
                )
            self.__representative[testset] = representative_per_class
        return self.__representative

    def write_underconfident(self):
        """
        Create folder (is not existent) and write the underconfident images
        """
        for testset in self.__test_datasets:
            testfodlers2create = os.path.join(self.folder2create, testset)
            df = self.__dataframes[testset]
            if not os.path.exists(testfodlers2create):
                os.makedirs(testfodlers2create)
            underconfident = self.__underconfimages[testset]
            underconfident.savefig(f"{testfodlers2create}/{testset}_underconfident.pdf")

    def write_overconfident(self):
        """
        Create folder (is not existent) and write the overconfident images
        """
        for testset in self.__test_datasets:
            testfodlers2create = os.path.join(self.folder2create, testset)
            df = self.__dataframes[testset]
            if not os.path.exists(testfodlers2create):
                os.makedirs(testfodlers2create)
            overconfident = self.__overconfimages[testset]
            overconfident.savefig(f"{testfodlers2create}/{testset}_overconfident.pdf")

    def write_representative(self):
        """
        Create folder (is not existent) and write the representative images
        """
        for testset in self.__test_datasets:
            testfodlers2create = os.path.join(self.folder2create, testset)
            if not os.path.exists(testfodlers2create):
                os.makedirs(testfodlers2create)
            pdfs_wo_failure = []
            for cla in self.__class2plot:
                name = self.__class2name[cla]
                repres_wo_failure = self.__representative[testset][cla]
                cluster_plots_folder = f"{testfodlers2create}/{name}"
                if not os.path.exists(cluster_plots_folder):
                    os.makedirs(cluster_plots_folder)

                path_repres_wo_failure = (
                    f"{cluster_plots_folder}/imgs_{name}_repres_wo_faillabel.pdf"
                )
                pdfs_wo_failure.append(path_repres_wo_failure)
                pdfs_represantative = []
                repres_wo_failure.savefig(path_repres_wo_failure)

                # merger = PdfMerger()
                # for pdf_3 in pdfs_wo_failure:
                #     merger.append(pdf_3)
                # merger.write(
                #     f"{testfodlers2create}/{testset}_imgs_represantative_wo_fail.pdf"
                # )
                # print(
                #     "Writing Represantiative to:",
                #     f"{testfodlers2create}/{testset}_imgs_represantative_wo_fail.pdf",
                # )
                #
                # merger.close()
                #
                # self.__encoderls[testset][cla].write_html(
                #     f"{cluster_plots_folder}/latentspace_{name}.html"
                # )
                # self.__encoderls[testset][cla].write_image(
                #     f"{cluster_plots_folder}/latentspace_{name}.png", scale=3
                # )

    def write_all(self):
        """
        Create folder (is not existent) and write the failures as well as representative images
        can just call all other write functions
        """
        self.write_overconfident()
        self.write_underconfident()
        self.write_representative()
