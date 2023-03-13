from visualfailureanalysis.utils.utils import (
    kmeans_cluster_representative_without_failurelabel,
)
from visualfailureanalysis.utils.utils import overconfident_images
from visualfailureanalysis.utils.utils import underconfident_images
from visualfailureanalysis.utils.utils import getdffromarrays

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
from numpy import random
from sklearn import metrics
from scipy.spatial import distance
import plotly.graph_objects as go
from collections import OrderedDict
import copy
import matplotlib
import plotly
import cv2
from typing import Tuple
from scipy.stats import mode
from flask import request
import warnings
from PyPDF2 import PdfMerger

warnings.filterwarnings(action="ignore", category=FutureWarning)
import os
import dash
from dash import dcc
from dash import html


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
        path: str,
        class2name: dict,
        class2plot: list[int],
        ls_testsets: list[str],
        test_datasets: list[str],
    ) -> None:

        self.__path = path + "/test_results/"
        self.__class2name = class2name
        self.__class2plot = class2plot
        self.__ls_testsets = ls_testsets
        self.__test_datasets = test_datasets
        # Loading Data from init parameters
        self.__raw_output = np.load(f"{self.__path}raw_output.npz")["arr_0"]
        self.__softmax_output = self.__raw_output[:, :-2]
        self.__out_class = np.argmax(np.squeeze(self.__softmax_output), axis=1)
        self.__labels = np.squeeze(np.asarray(self.__raw_output[:, -2]))
        self.__dataset_idx = np.squeeze(np.asarray(self.__raw_output[:, -1]))
        self.__encoded_output = np.load(f"{self.__path}encoded_output.npz")["arr_0"]
        self.__test_datasets_length = len(self.__ls_testsets)
        self.__csvs = []
        self.__dataframes = {}
        self.__encoderls = {}
        self.accuracies_dict = {}
        for i in range(int(self.__test_datasets_length)):
            try:
                attributions = pd.read_csv(f"{self.__path}attributions{i}.csv")
            except:
                attributions = pd.read_csv(f"{self.__path}attributions.csv")
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

    def print_summary_stats(self):
        """
        Returns some object with statistics and prints in consol
        """
        ### loop over datasets
        for i in range(int(self.__test_datasets_length)):
            accuracies_per_class = {}
            study = self.__ls_testsets[i]
            self.__boolarray = self.__encoded_output[:, -1] == i
            len_testset = np.sum(self.__boolarray)
            predicted = self.__out_class[self.__boolarray]
            baseline_class = mode(self.__labels)[0][0]
            encoded = self.__encoded_output[self.__boolarray][:, :-1]
            softmax = self.__softmax_beginning[self.__boolarray]
            y_score = self.__softmax_beginning[self.__boolarray, 1]
            y_true = self.__labels[self.__boolarray]
            # carefull: order
            label_names_ls = [x for _, x in self.__class2name.items()]
            ###Basic Classifier metrics
            if self.n_classes == 2:
                df_classifier_metrics = pd.DataFrame()
                acc = accuracy_score(y_true=y_true, y_pred=predicted)
                try:
                    auc = roc_auc_score(y_true=y_true, y_score=y_score)
                    ap = average_precision_score(y_true=y_true, y_score=y_score)
                except:
                    print("Only 1 class in y_true")
                    auc = None
                    ap = None
                best_baseline = np.ones((sum(self.__boolarray))) * baseline_class
                baseline_acc = accuracy_score(y_true=y_true, y_pred=best_baseline)
                print(f"{auc=},{ap=},{acc=},{baseline_acc=}")
                df_classifier_metrics["AUC"] = auc
                df_classifier_metrics["AP"] = ap
                df_classifier_metrics["Accuracy"] = acc
                df_classifier_metrics["Baseline Accuracy"] = baseline_acc
                for cla in range(self.n_classes):
                    bool_cla = y_true == cla
                    y_true_cla = y_true[bool_cla]
                    predicted_cla = predicted[bool_cla]
                    acc_cla = accuracy_score(y_true=y_true_cla, y_pred=predicted_cla)
                    ratio_class = sum(bool_cla) / len(bool_cla)
                    print(f"\t{cla=}:{acc_cla=}, {ratio_class=}")
                    accuracies_per_class[cla] = acc_cla
            if self.n_classes > 2:
                acc = accuracy_score(y_true=y_true, y_pred=predicted)
                best_baseline = np.ones((sum(self.__boolarray))) * baseline_class
                baseline_acc = accuracy_score(y_true=y_true, y_pred=best_baseline)
                print(f"total:{acc=},{baseline_acc=}")
                for cla in range(self.n_classes):
                    bool_cla = y_true == cla
                    y_true_cla = y_true[bool_cla]
                    predicted_cla = predicted[bool_cla]
                    acc_cla = accuracy_score(y_true=y_true_cla, y_pred=predicted_cla)
                    ratio_class = sum(bool_cla) / len(bool_cla)
                    print(f"\t{cla=}:{acc_cla=}, {ratio_class=}")
                    accuracies_per_class[cla] = acc_cla
            self.accuracies_dict[study] = accuracies_per_class

    def setup(self):
        """
        Run T-SNE and build data frame for further analysis
        """
        self.print_summary_stats()
        # might take a while because of TSNE
        for testset in self.__test_datasets:
            i = self.__ls_testsets.index(testset)
            study = self.__ls_testsets[i]
            boolarray = self.__encoded_output[:, -1] == i
            len_testset = np.sum(boolarray)
            predicted = self.__out_class[boolarray]
            baseline_class = mode(self.__labels)[0][0]
            encoded = self.__encoded_output[boolarray][:, :-1]
            softmax = self.__softmax_beginning[boolarray]
            y_score = self.__softmax_beginning[boolarray, 1]
            y_true = self.__labels[boolarray]
            subfolders = self.__path.split("/")
            cwd = os.getcwd()
            dir = subfolders[-4:-2]
            folder2create = os.path.join(cwd, "outputs", dir[0], dir[1])
            print(folder2create)
            self.folder2create = folder2create
            ###Encoder plots
            check_file = folder2create + "/" + study + "/dataframe.csv"
            if os.path.exists(check_file):
                df = pd.read_csv(check_file)
                self.__dataframes[testset] = df
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
                self.__dataframes[testset] = df

            ###create encoder plot per class
            xmax = np.max(df["0"])
            xmin = np.min(df["0"])
            ymax = np.max(df["1"])
            ymin = np.min(df["1"])
            zmax = np.max(df["2"])
            zmin = np.min(df["2"])
            figures = {}
            for cla in self.__class2plot:
                fig = go.Figure()
                for cf in ["TP", "TN", "FP", "FN"]:
                    if cf == "TP":
                        sub_df = df[(df.label == cla) & (df.predicted == cla)]
                    if cf == "TN":
                        sub_df = df[(df.label != cla) & (df.predicted != cla)]
                    if cf == "FP":
                        sub_df = df[(df.label != cla) & (df.predicted == cla)]
                    if cf == "FN":
                        sub_df = df[(df.label == cla) & (df.predicted != cla)]

                    fig.add_trace(
                        go.Scatter3d(
                            x=sub_df["0"],
                            y=sub_df["1"],
                            z=sub_df["2"],
                            opacity=0.6,
                            mode="markers",
                            name=cf,
                            text=sub_df["filepath"],
                            marker=dict(
                                size=3, color=self.color_map[cf], symbol="circle"
                            ),
                        )
                    )
                fig.update_layout(
                    coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside")
                )
                fig.layout.hovermode = "closest"
                fig.update_layout(width=1000, height=1000, template="simple_white")
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            range=[xmin, xmax],
                        ),
                        yaxis=dict(
                            range=[ymin, ymax],
                        ),
                        zaxis=dict(
                            range=[zmin, zmax],
                        ),
                    )
                )
                camera = dict(eye=dict(x=1.4, y=1.4, z=1.4))
                font = dict(size=16)
                fig.update_layout(
                    scene_camera=camera, font=font, legend=dict(font=dict(size=18))
                )
                figures[cla] = fig
            self.__encoderls[testset] = figures
            testfodlers2create = os.path.join(folder2create, testset)
            print(testfodlers2create)
            if not os.path.exists(testfodlers2create):
                os.makedirs(testfodlers2create)
            self.__dataframes[testset].to_csv(
                testfodlers2create + "/dataframe.csv", index=False
            )
        self.show_underconfident()
        self.show_overconfident()
        self.show_representative()

    def prepair_dash(self):
        """
        Create folder (is not existent) and write the data frame for the dash app
        """
        for testset in self.__test_datasets:
            subfolders = self.__path.split("/")
            cwd = os.getcwd()
            dir = subfolders[-4:-2]
            dash_fodler = os.path.join(cwd, "dash_data")
            self.dash_fodler = dash_fodler
            if not os.path.exists(dash_fodler):
                os.makedirs(dash_fodler)
            self.__dataframes[testset].to_csv(
                dash_fodler + "/" + dir[0] + dir[1] + testset + ".csv", index=False
            )

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

                merger = PdfMerger()
                for pdf_3 in pdfs_wo_failure:
                    merger.append(pdf_3)
                merger.write(
                    f"{testfodlers2create}/{testset}_imgs_represantative_wo_fail.pdf"
                )
                print(
                    "Writing Represantiative to:",
                    f"{testfodlers2create}/{testset}_imgs_represantative_wo_fail.pdf",
                )

                merger.close()

                self.__encoderls[testset][cla].write_html(
                    f"{cluster_plots_folder}/latentspace_{name}.html"
                )
                self.__encoderls[testset][cla].write_image(
                    f"{cluster_plots_folder}/latentspace_{name}.png", scale=3
                )

    def write_all(self):
        """
        Create folder (is not existent) and write the failures as well as representative images
        can just call all other write functions
        """
        self.write_overconfident()
        self.write_underconfident()
        self.write_representative()
