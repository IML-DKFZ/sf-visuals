import copy
import warnings
from collections import OrderedDict

import cv2
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)

warnings.filterwarnings(action="ignore", category=FutureWarning)
import os

matplotlib.use("agg")

color_map = {
    "TN": "rgb(26,150,65)",
    "FN": "rgb(166,217,106)",
    "FP": "rgb(253,174,97)",
    "TP": "rgb(215,25,28)",
}
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


def agg_df_for_app():
    x = glob.glob("./outputs/*/dataframe.csv")
    cwd = os.getcwd()
    for path in x:
        src = copy.copy(path)
        folders = []
        while True:
            path, folder = os.path.split(path)
            if folder != "":
                folders.append(folder)
            elif path != "":
                folders.append(path)
                break
        folders.reverse()
        dst = cwd + "/dash_data/" + folders[-3] + "_" + folders[-2] + "_" + folders[-1]
        shutil.copy(src, dst)


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

    columns = 3
    rows = 3

    sub_df = df[df.label == cla]

    data = sub_df[["0", "1", "2"]].to_numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

    prediction = kmeans.predict(data)
    centers = kmeans.cluster_centers_

    fig3 = plt.figure(figsize=(16, 16))

    for j in range(n_clusters):
        try:
            center = centers[j, :]
        except:
            center = centers[0, :]
        ids = closest_node(center, data, k=1)
        i = j + 1
        fig3.add_subplot(rows, columns, i)
        file = sub_df["filepath"].iloc[ids[0]]
        # for cluster datapath in fp
        print(file)
        start, end = file.split("levin/")
        # file = "/home/t974t/Data/levin/" + end
        try:
            im = cv2.imread(file)
            RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except:
            file = "/home/t974t/NetworkDrives/E130-Personal/Kobelke/" + end
            im = cv2.imread(file)
            RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = sub_df["label"].iloc[ids[0]]
        label = int(label)
        predicted = sub_df["predicted"].iloc[ids[0]]
        # try:
        #    subclass = sub_df["dx"].iloc[ids[0]]
        #    name = f"label: {class2name[label]}, pred: {class2name[predicted]}, subclass: {subclass}"
        # except:
        #    name = f"label: {class2name[label]}, pred: {class2name[predicted]}"

        # plt.title(name)
        plt.imshow(RGB_im)
        plt.axis("off")
        plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)
    # fig3.suptitle(f"Class Accuracy: {cla_accuracies[cla]:.2f}", fontsize=70)
    return fig3


def overconfident_images(df, class2name):
    df["confid"] = -1
    # placeholder
    for i in range(len(df)):
        row = df.iloc[i]
        pred = int(row.predicted)
        confid = row[f"softmax{pred}"]
        df.at[i, "confid"] = confid
    df_sub = df[~(df.label == df.predicted)]
    df_oc_first = df_sub.sort_values(by="confid", ascending=False, ignore_index=True)

    fig = plt.figure(figsize=(16, 16))
    columns = 1
    rows = 3
    k = columns * rows

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i == 2:
            i = 6  # avoid duplicat in isic
        if i == 3:
            i = 14  # avoid duplicat in isic

        file = df_oc_first.filepath[i - 1]
        # for cluster images
        start, end = file.split("l049e/")
        file = "/home/l049e/Data/" + end
        label = df_oc_first.label[i - 1]
        label = int(label)
        pred = df_oc_first.predicted[i - 1]
        lab_pred = f"{label=}, {pred=}"
        print(file)
        try:
            im = cv2.imread(file)
            RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except:
            file = "/home/l049e/E130-Personal/Kobelke/" + end
            im = cv2.imread(file)
            RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_im)
        conf = df_oc_first.confid[i - 1]
        # if lab==1: conf = -1 * conf
        # if lab==-1: conf = 1 - conf
        conf = np.round(conf, decimals=2)
        font = {"family": "DejaVu Sans", "weight": "bold", "size": 22}

        matplotlib.rc("font", **font)
        plt.axis("off")
        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(wspace=0.0001)
        title = f"L: {class2name[label]} \nP: {class2name[pred]} \nC: {conf:.2f}"
        plt.title(title, loc="left")

    return fig


def underconfident_images(df, class2name):
    df["confid"] = -1
    # placeholder
    for i in range(len(df)):
        row = df.iloc[i]
        pred = int(row.predicted)
        confid = row[f"softmax{pred}"]
        df.at[i, "confid"] = confid
    df_sub = df[(df.label == df.predicted)]
    df_oc_first = df_sub.sort_values(by="confid", ascending=True, ignore_index=True)

    fig = plt.figure(figsize=(16, 16))
    columns = 1
    rows = 3
    k = columns * rows

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i == 2:
            i = 6  # avoid duplicat in isic
        if i == 3:
            i = 14  # avoid duplicat in isic
        file = df_oc_first.filepath[i - 1]
        # for cluster images
        start, end = file.split("l049e/")
        file = "/home/l049e/Data/" + end
        label = df_oc_first.label[i - 1]
        label = int(label)
        pred = df_oc_first.predicted[i - 1]
        lab_pred = f"{label=}, {pred=}"
        print(file)
        try:
            im = cv2.imread(file)
            RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except:
            file = "/home/l049e/E130-Personal/Kobelke/" + end
            im = cv2.imread(file)
            RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_im)
        conf = df_oc_first.confid[i - 1]
        # if lab==1: conf = -1 * conf
        # if lab==-1: conf = 1 - conf
        conf = np.round(conf, decimals=2)
        font = {"family": "DejaVu Sans", "weight": "bold", "size": 22}

        matplotlib.rc("font", **font)
        plt.axis("off")
        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(wspace=0.0001)
        title = f"L: {class2name[label]} \nP: {class2name[pred]} \nC: {conf:.2f}"
        plt.title(title, loc="left")

    return fig


def performe_visual_analysis(
    path: str,
    class2name: OrderedDict,
    class2plot: list[int],
    test_datasets: list[str],
    ls_testsets: list[str],
    domain: str,
):
    path = path + "/test_results/"
    raw_output = np.load(f"{path}raw_output.npz")
    raw_output = raw_output["arr_0"]

    softmax_output = raw_output[:, :-2]
    out_class = np.argmax(np.squeeze(softmax_output), axis=1)

    labels = raw_output[:, -2]
    labels = np.squeeze(np.asarray(labels))

    dataset_idx = raw_output[:, -1]
    dataset_idx = np.squeeze(np.asarray(dataset_idx))
    encoded_output = np.load(f"{path}encoded_output.npz")
    encoded_output = encoded_output["arr_0"]
    test_datasets_length = len(ls_testsets)
    csvs = []
    confls = []
    dataframes = {}
    histogramsls = []
    encoderls = {}
    accuracies_dict = {}
    for i in range(int(test_datasets_length)):
        try:
            attributions = pd.read_csv(f"{path}attributions{i}.csv")
        except:
            attributions = pd.read_csv(f"{path}attributions.csv")
        csvs.append(attributions)
    softmax_beginning = np.squeeze(softmax_output)
    n_classes = int(np.max(labels) + 1)
    if n_classes <= 1:
        raise ValueError("Only 1 class in label output or negative label!")

    ### loop over datasets
    for i in range(int(test_datasets_length)):
        accuracies_per_class = {}
        study = ls_testsets[i]
        boolarray = encoded_output[:, -1] == i
        len_testset = np.sum(boolarray)
        len_testset = np.sum(boolarray)
        predicted = out_class[boolarray]
        baseline_class = mode(labels)[0][0]
        encoded = encoded_output[boolarray][:, :-1]
        softmax = softmax_beginning[boolarray]
        y_score = softmax_beginning[boolarray, 1]
        y_true = labels[boolarray]
        # carefull: order
        label_names_ls = [x for _, x in class2name.items()]
        ###Basic Classifier metrics
        if True:
            if n_classes == 2:
                df_classifier_metrics = pd.DataFrame()
                acc = accuracy_score(y_true=y_true, y_pred=predicted)
                try:
                    auc = roc_auc_score(y_true=y_true, y_score=y_score)
                    ap = average_precision_score(y_true=y_true, y_score=y_score)
                except:
                    print("Only 1 class in y_true")
                    auc = None
                    ap = None
                best_baseline = np.ones((sum(boolarray))) * baseline_class
                baseline_acc = accuracy_score(y_true=y_true, y_pred=best_baseline)
                print(f"{auc=},{ap=},{acc=},{baseline_acc=}")
                df_classifier_metrics["AUC"] = auc
                df_classifier_metrics["AP"] = ap
                df_classifier_metrics["Accuracy"] = acc
                df_classifier_metrics["Baseline Accuracy"] = baseline_acc
                for cla in range(n_classes):
                    bool_cla = y_true == cla
                    y_true_cla = y_true[bool_cla]
                    predicted_cla = predicted[bool_cla]
                    acc_cla = accuracy_score(y_true=y_true_cla, y_pred=predicted_cla)
                    ratio_class = sum(bool_cla) / len(bool_cla)
                    print(f"\t{cla=}:{acc_cla=}, {ratio_class=}")
                    accuracies_per_class[cla] = acc_cla
            if n_classes > 2:
                acc = accuracy_score(y_true=y_true, y_pred=predicted)
                best_baseline = np.ones((sum(boolarray))) * baseline_class
                baseline_acc = accuracy_score(y_true=y_true, y_pred=best_baseline)
                print(f"total:{acc=},{baseline_acc=}")
                for cla in range(n_classes):
                    bool_cla = y_true == cla
                    y_true_cla = y_true[bool_cla]
                    predicted_cla = predicted[bool_cla]
                    acc_cla = accuracy_score(y_true=y_true_cla, y_pred=predicted_cla)
                    ratio_class = sum(bool_cla) / len(bool_cla)
                    print(f"\t{cla=}:{acc_cla=}, {ratio_class=}")
                    accuracies_per_class[cla] = acc_cla
        accuracies_dict[study] = accuracies_per_class

    centers_testset = {}
    cluster_plots = {}

    # might take a while because of TSNE
    for testset in test_datasets:
        i = ls_testsets.index(testset)
        study = ls_testsets[i]
        boolarray = encoded_output[:, -1] == i
        len_testset = np.sum(boolarray)
        predicted = out_class[boolarray]
        baseline_class = mode(labels)[0][0]
        encoded = encoded_output[boolarray][:, :-1]
        softmax = softmax_beginning[boolarray]
        y_score = softmax_beginning[boolarray, 1]
        y_true = labels[boolarray]
        subfolders = path.split("/")
        cwd = os.getcwd()
        dir = subfolders[-4:-2]
        folder2create = os.path.join(cwd, "outputs", dir[0], dir[1])
        print(folder2create)

        ###Encoder plots
        check_file = folder2create + "/" + study + "/dataframe.csv"
        if os.path.exists(check_file):
            df = pd.read_csv(check_file)
            dataframes[testset] = df
        else:
            pca50 = PCA(n_components=50)
            pca_encoded50 = pca50.fit_transform(encoded)
            tsne_encoded3_set = TSNE(
                n_components=3, init="pca", learning_rate=200
            ).fit_transform(pca_encoded50)
            ### create dataframe for each dataset that stores all relevant information
            testset_csv = csvs[i]
            fp = testset_csv.filepath
            ### savety_ check that y_true from raw output == targets from the csv to ensure correct ordering/dataset
            df = getdffromarrays(
                labels=y_true,
                out_class=predicted,
                tsne_encoded3=tsne_encoded3_set,
                softmax=softmax,
                filepath=fp,
            )
            dataframes[testset] = df

        ###create encoder plot per class
        xmax = np.max(df["0"])
        xmin = np.min(df["0"])
        ymax = np.max(df["1"])
        ymin = np.min(df["1"])
        zmax = np.max(df["2"])
        zmin = np.min(df["2"])
        figures = {}
        for cla in class2plot:
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
                        marker=dict(size=3, color=color_map[cf], symbol="circle"),
                    )
                )
            fig.update_layout(
                coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside")
            )
            fig = go.FigureWidget(fig.data, fig.layout)
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
        encoderls[testset] = figures
        testfodlers2create = os.path.join(folder2create, testset)
        print(testfodlers2create)
        if not os.path.exists(testfodlers2create):
            os.makedirs(testfodlers2create)
        df_save = dataframes[testset].to_csv(
            testfodlers2create + "/dataframe.csv", index=False
        )
    k = 9
    # for testset in test_datasets:
    #    test_idx = ls_testsets.index(testset)
    #    print(testset)
    #    from sklearn.cluster import KMeans
    #
    #    n_clusters = k
    #    # test_idx = 1
    #    df = dataframes[testset]
    #    boolarray = encoded_output[:, -1] == test_idx
    #    encoded = encoded_output[boolarray][:, :-1]
    #
    #    cluster_cla = {}
    #    centers_cla = {}
    #
    #    for cla in class2plot:
    #        cluster_sub = {}
    #        center_sub = {}
    #
    #        for cf in ["TP", "TN", "FP", "FN"]:
    #
    #            if cf == "TP":
    #                sub_df = df[(df.label == cla) & (df.predicted == cla)]
    #            if cf == "TN":
    #                sub_df = df[(df.label != cla) & (df.predicted != cla)]
    #            if cf == "FP":
    #                sub_df = df[(df.label != cla) & (df.predicted == cla)]
    #            if cf == "FN":
    #                sub_df = df[(df.label == cla) & (df.predicted != cla)]
    #            pca_enc = sub_df[["0", "1", "2"]].to_numpy()
    #            try:
    #                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_enc)
    #            except:
    #                n_clusters = int(len(pca_enc))
    #                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_enc)
    #            prediction = kmeans.predict(pca_enc)
    #            centers = kmeans.cluster_centers_
    #
    #            fig = go.Figure()
    #            for i in range(n_clusters):
    #                df_sub = sub_df[prediction == i]
    #                fig.add_trace(
    #                    go.Scatter3d(
    #                        x=df_sub["0"],
    #                        y=df_sub["1"],
    #                        z=df_sub["2"],
    #                        opacity=0.6,
    #                        mode="markers",
    #                        marker=dict(size=3, color=colors[i], symbol="circle"),
    #                    )
    #                )
    #                fig.add_trace(
    #                    go.Scatter3d(
    #                        x=[centers[i, 0]],
    #                        y=[centers[i, 1]],
    #                        z=[centers[i, 2]],
    #                        opacity=0.8,
    #                        mode="markers",
    #                        marker=dict(size=7, color=colors[i], symbol="circle"),
    #                    )
    #                )
    #            fig.update_layout(width=1000, height=1000, template="simple_white")
    #            fig.update_layout(
    #                scene=dict(
    #                    xaxis=dict(
    #                        range=[xmin, xmax],
    #                    ),
    #                    yaxis=dict(
    #                        range=[ymin, ymax],
    #                    ),
    #                    zaxis=dict(
    #                        range=[zmin, zmax],
    #                    ),
    #                )
    #            )
    #            camera = dict(eye=dict(x=1.4, y=1.4, z=1.4))
    #            font = dict(size=16)
    #            fig.update_layout(
    #                scene_camera=camera, font=font, legend=dict(font=dict(size=18))
    #            )
    #            # fig.show()
    #            cluster_sub[cf] = fig
    #            center_sub[cf] = centers
    #
    #        cluster_cla[cla] = cluster_sub
    #        centers_cla[cla] = center_sub
    #
    #    cluster_plots[testset] = cluster_cla
    #    centers_testset[testset] = centers_cla

    from PyPDF2 import PdfMerger

    for testset in test_datasets:
        testfodlers2create = os.path.join(folder2create, testset)
        df = dataframes[testset]
        if not os.path.exists(testfodlers2create):
            os.makedirs(testfodlers2create)
        pdfs_wo_failure = []
        for cla in class2plot:
            name = class2name[cla]
            # cluster_images = kmeans_cluster_images(
            #    dataframe=df,
            #    cla=cla,
            #    k=k,
            #    centers_dict=centers_testset[testset][cla],
            #    class2name=class2name,
            # )
            # cluster_images_represantative= kmeans_cluster_representative(dataframe=df, cla=cla,k=k, centers_dict=centers_testset[testset][cla], class2name=class2name)
            repres_wo_failure = kmeans_cluster_representative_without_failurelabel(
                cla_accuracies=accuracies_dict[testset],
                dataframe=df,
                cla=cla,
                class2name=class2name,
            )
            cluster_plots_folder = f"{testfodlers2create}/{name}"
            if not os.path.exists(cluster_plots_folder):
                os.makedirs(cluster_plots_folder)
            overconfident = overconfident_images(df=df)
            underconfident = underconfident_images(df=df)
            overconfident.savefig(
                f"{testfodlers2create}/{domain}_{testset}_overconfident.pdf"
            )
            print(
                "Writing Failure Overconf to:",
                f"{testfodlers2create}/{domain}_{testset}_overconfident.pdf",
            )

            underconfident.savefig(
                f"{testfodlers2create}/{domain}_{testset}_underconfident.pdf"
            )
            print(
                "Writing Failure Underconf to:",
                f"{testfodlers2create}/{domain}_{testset}_underconfident.pdf",
            )

            path_repres_wo_failure = (
                f"{cluster_plots_folder}/imgs_{name}_repres_wo_faillabel.pdf"
            )
            pdfs_wo_failure.append(path_repres_wo_failure)
            pdfs_represantative = []
            repres_wo_failure.savefig(path_repres_wo_failure)

            # for cf in cluster_images.keys():
            #    pdfs = []
            #    file_path_represantative = (
            #        f"{cluster_plots_folder}/imgs_{name}_{cf}_repres.pdf"
            #    )
            #    # cluster_images_represantative[cf][0].savefig(file_path_represantative)
            #    pdfs_represantative.append(file_path_represantative)
            #    for cluster_idx in range(k):
            #        file_path_plot = (
            #            f"{cluster_plots_folder}/imgs_{name}_{cf}_{cluster_idx}.pdf"
            #        )
            #        cluster_images[cf][cluster_idx].savefig(file_path_plot)
            #        pdfs.append(file_path_plot)
            #    merger = PdfMerger()
            #    for pdf in pdfs:
            #        merger.append(pdf)
            #    merger.write(f"{cluster_plots_folder}/imgs_{name}_{cf}_plots.pdf")
            #    merger.close()
            #
            #    cluster_plots[testset][cla][cf].write_html(
            #        f"{cluster_plots_folder}/cluster_{name}_{cf}.html"
            #    )
            #    cluster_plots[testset][cla][cf].write_image(
            #        f"{cluster_plots_folder}/cluster_{name}_{cf}.png", scale=3
            #    )
            merger = PdfMerger()
            # for pdf_2 in pdfs_represantative:
            #    merger.append(pdf_2)
            # merger.write(f"{cluster_plots_folder}/imgs_{name}_represantative.pdf")
            # merger.close()
            # merger = PdfMerger()
            for pdf_3 in pdfs_wo_failure:
                merger.append(pdf_3)
            merger.write(
                f"{testfodlers2create}/{domain}_{testset}_imgs_represantative_wo_fail.pdf"
            )
            print(
                "Writing Represantiative to:",
                f"{testfodlers2create}/{domain}_{testset}_imgs_represantative_wo_fail.pdf",
            )

            merger.close()

            encoderls[testset][cla].write_html(
                f"{cluster_plots_folder}/latentspace_{name}.html"
            )
            encoderls[testset][cla].write_image(
                f"{cluster_plots_folder}/latentspace_{name}.png", scale=3
            )
