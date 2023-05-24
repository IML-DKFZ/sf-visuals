import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans

warnings.filterwarnings(action="ignore", category=FutureWarning)
import os

matplotlib.use("agg")

colors = [
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


def closest_node(node, nodes, k: int):
    distances = distance.cdist([node], nodes)
    idx = np.argpartition(distances, k)
    return idx[0][0:k]


def getdffromarrays(
    labels, out_class, tsne_encoded3, softmax, filepath: str
) -> pd.DataFrame:
    df = pd.DataFrame(data=tsne_encoded3)
    df["label"] = labels
    df["predicted"] = out_class
    _, n = softmax.shape
    for i in range(n):
        df[f"softmax{i}"] = softmax[:, i]
    df["filepath"] = filepath
    df.filepath = df.filepath.str.replace(
        "/dkfz/cluster/gpu/data/OE0612/l049e", "/home/t974t/Data/levin"
    )
    df["0"] = df[0]
    df["1"] = df[1]
    df["2"] = df[2]
    df = df.drop(labels=[0, 1, 2], axis=1)
    return df


def kmeans_cluster_representative_without_failurelabel(
    dataframe, cla: int  # , class2name: dict, cla_accuracies: dict
) -> plt.Figure:
    k = 9
    n_clusters = k
    df = dataframe
    df.filepath = df.filepath.str.replace(
        "/dkfz/cluster/gpu/data/OE0612/l049e", "/home/t974t/Data/levin"
    )

    sub_df = df[df.label == cla]

    data = sub_df[["0", "1", "2"]].to_numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

    kmeans.predict(data)
    centers = kmeans.cluster_centers_

    idx = []
    for j in range(n_clusters):
        try:
            center = centers[j, :]
        except:
            center = centers[0, :]
        ids = closest_node(center, data, k=1)
        idx.append(ids[0])
        # logger.info("Reading representative image {}", file)

    return sub_df.iloc[idx]
