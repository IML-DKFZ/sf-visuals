from __future__ import annotations

import argparse
import base64
import itertools
from dataclasses import dataclass
from pathlib import Path

import dash_daq as daq
import plotly.graph_objects as go
from dash import ALL, MATCH, Dash, Input, Output, Patch, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from loguru import logger

from sf_visuals.analyser import Analyser


@dataclass
class AppState:
    base_path: Path
    path: Path
    analyser: Analyser


def _tab_latent_space():
    return dcc.Tab(
        label="Latent Space",
        children=html.Div(
            id="tab-latentspace",
            className="tab-custom",
            children=[
                dcc.Loading(
                    html.Div(
                        [
                            dcc.Graph(
                                id="latentspace",
                                className="latentspace-inner",
                                responsive=True,
                                clear_on_unhover=True,
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P("Marker Size"),
                                            daq.Slider(
                                                id="marker-size",
                                                value=5,
                                                min=1,
                                                max=20,
                                                handleLabel={
                                                    "showCurrentValue": True,
                                                    "label": "value",
                                                },
                                            ),
                                        ],
                                        className="slider-container",
                                    ),
                                    html.Div(
                                        [
                                            html.P("Marker Alpha"),
                                            daq.Slider(
                                                id="marker-alpha",
                                                value=0.6,
                                                min=0.1,
                                                max=1.0,
                                                step=0.1,
                                                handleLabel={
                                                    "showCurrentValue": True,
                                                    "label": "value",
                                                },
                                            ),
                                        ],
                                        className="slider-container",
                                    ),
                                ],
                                className="sliders",
                            ),
                        ],
                        className="latentspace",
                    ),
                    className="latentspace-loading",
                ),
                html.Div(
                    dcc.Loading(
                        html.Div(
                            id="hover-preview",
                        ),
                    ),
                    className="hover-preview",
                ),
                html.Div(children=[], hidden=True, id="dummy"),
            ],
        ),
    )


def _tab_failures():
    return dcc.Tab(
        label="Failures",
        children=html.Div(
            id="tab-failures",
            className="tab-custom",
            children=[
                dcc.Loading(
                    html.Div(
                        id="representative-view",
                        className="representative-view",
                    ),
                ),
                dcc.Loading(
                    html.Div(
                        id="failure-view",
                        className="failure-view",
                    ),
                ),
            ],
        ),
    )


def _sidebar_folder_selection(app_state: AppState):
    all_paths = sorted(
        [
            str(path.parent.relative_to(app_state.base_path))
            for path in app_state.base_path.glob("**/test_results")
        ]
    )

    return [
        html.H2("Experiment folder:"),
        dcc.Dropdown(
            all_paths,
            str(app_state.path.relative_to(app_state.base_path)),
            id="base-path-dd",
            clearable=False,
        ),
        html.Div(id="path-display-dd"),
    ]


def _sidebar_class_selection(app_state: AppState):
    return [
        html.H3("Classes:"),
        dcc.Checklist(
            [
                {
                    "label": html.Div(
                        [
                            f"Class {c}: ",
                            dcc.Input(
                                id={
                                    "type": "class-name",
                                    "id": f"{c}",
                                },
                                type="text",
                                value=f"{c}",
                            ),
                        ],
                        style={"display": "inline-block"},
                    ),
                    "value": c,
                }
                for c in app_state.analyser.classes
            ],
            app_state.analyser.classes,
            id="checklist-classes",
        ),
    ]


def _sidebar_dataset_selection(app_state: AppState):
    return [
        html.H3("Datasets:"),
        dcc.Checklist(
            ["iid", "ood"],
            ["iid", "ood"],
            id="checklist-testsets",
        ),
        html.Div(
            dcc.RadioItems(
                [
                    {
                        "label": html.Div(
                            [f"Testset {c}"], style={"display": "inline-block"}
                        ),
                        "value": c,
                    }
                    for c in app_state.analyser.testsets
                    if c != "iid"
                ],
                [c for c in app_state.analyser.testsets if c != "iid"][0],
                id="selection-testset",
            ),
            id="container-selection-testset",
        ),
    ]


def _sidebar_color_selection(app_state: AppState):
    return [
        html.H3(
            children="Color By:",
        ),
        dcc.RadioItems(
            [
                {"label": "Confidence", "value": "confidence"},
                {"label": "Source/Target", "value": "source-target"},
                {"label": "Class Confusion", "value": "class-confusion"},
            ],
            "confidence",
            id="checklist-colorby",
        ),
    ]


def _sidebar(app_state: AppState):
    return html.Div(
        id="sidebar",
        style={"display": "inline-block"},
        children=(
            _sidebar_folder_selection(app_state)
            + _sidebar_class_selection(app_state)
            + _sidebar_dataset_selection(app_state)
            + _sidebar_color_selection(app_state)
        ),
    )


def _failure_triplet(testset: str, stats: list[dict]):
    imgs = []
    for stat in stats:
        imgpath = stat["filepath"].replace(
            "/dkfz/cluster/gpu/data/OE0612/l049e/", "/home/t974t/Data/levin/"
        )
        with open(imgpath, "rb") as img:
            data = base64.b64encode(img.read()).replace(b"\n", b"").decode("utf-8")

        imgs.append(
            html.Div(
                [
                    html.Img(
                        id={"type": "failure-img", "id": imgpath, "testset": testset},
                        className="failure-img",
                        height="128px",
                        src=f"data:image/png;base64,{data}",
                        n_clicks=0,
                    ),
                    html.Div(
                        [
                            html.P(f"Pr: {stat['predicted']}"),
                            html.P(f"GT: {stat['label']}"),
                            html.P(f"C : {stat['confid']}"),
                        ],
                        className="failure-stat",
                    ),
                ],
                className="failure-img-container",
            )
        )
    return html.Div(
        children=[
            html.H5(f"Testset: {testset}"),
            html.Div(
                imgs,
                className="failure-container",
            ),
        ]
    )


def main():
    app = Dash(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()

    app_state = AppState(
        base_path=Path(
            "/home/t974t/NetworkDrives/E130-Personal/Kobelke/cluster_checkpoints/"
        ),
        path=args.path,
        analyser=Analyser(path=args.path),
    )

    app.layout = html.Div(
        className="app-container",
        children=[
            _sidebar(app_state),
            dcc.Tabs(
                parent_className="container",
                className="tab-container",
                children=[_tab_latent_space(), _tab_failures()],
            ),
        ],
    )

    @app.callback(
        Output("dummy", "children"),
        Input({"type": "class-name", "id": ALL}, "value"),
        State({"type": "class-name", "id": ALL}, "id"),
    )
    def update_class_name(value, id):
        classes = {}
        for i, v in zip(id, value):
            classes[int(i["id"])] = v
        app_state.analyser.classes = classes

    @app.callback(
        Output("latentspace", "figure"),
        Input("checklist-testsets", "value"),
        Input("selection-testset", "value"),
        Input("checklist-classes", "value"),
        Input("checklist-colorby", "value"),
    )
    def update_testset(iid_ood, testset: str, classes, colorby):
        testsets = []
        if "iid" in iid_ood:
            testsets.append("iid")
        if "ood" in iid_ood:
            testsets.append(testset)

        if len(testsets) == 0:
            return None

        logger.info("Testsets to display: {}", testsets)

        figure = app_state.analyser.plot_latentspace(
            tuple(testsets), classes2plot=tuple(classes), coloring=colorby
        )

        figure["layout"]["uirevision"] = True

        return figure

    @app.callback(
        Output("representative-view", "children"),
        Input("base-path-dd", "value"),
        Input("checklist-classes", "value"),
        Input("selection-testset", "value"),
        Input("checklist-testsets", "value"),
    )
    def update_testset2(base_path, classes, testset, iid_ood):
        testsets = []
        if "iid" in iid_ood:
            testsets.append("iid")
        if "ood" in iid_ood:
            testsets.append(testset)

        if len(testsets) == 0:
            return None
        logger.info("Testsets to display: {}", testsets)

        imgs = []
        for testset, cls in itertools.product(testsets, classes):
            svg = app_state.analyser.representative(testset, cls)
            data = base64.b64encode(svg).replace(b"\n", b"").decode("utf-8")
            imgs.append(
                html.Div(
                    children=[
                        html.H5(f"Testset: {testset}, Class: {cls}"),
                        html.Img(
                            id="curimg",
                            width="512px",
                            height="512px",
                            src=f"data:image/svg;base64,{data}",
                        ),
                    ]
                )
            )

        return imgs

    @app.callback(
        Output("failure-view", "children"),
        Input("base-path-dd", "value"),
        Input("checklist-classes", "value"),
        Input("selection-testset", "value"),
        Input("checklist-testsets", "value"),
    )
    def update_testset3(base_path, classes, testset, iid_ood):
        testsets = []
        if "iid" in iid_ood:
            testsets.append("iid")
        if "ood" in iid_ood:
            testsets.append(testset)

        if len(testsets) == 0:
            return None
        logger.info("Testsets to display: {}", testsets)

        imgs = []
        for testset in testsets:
            stats = app_state.analyser.overconfident(testset)
            imgs.append(_failure_triplet(testset, stats))

        return imgs

    @app.callback(
        Output("hover-preview", "children"),
        Input("latentspace", "hoverData"),
        Input("latentspace", "clickData"),
    )
    def update_on_hover(hoverData, clickData):
        if hoverData is None:
            if clickData is None:
                raise PreventUpdate
            hoverData = clickData

        imgpath = hoverData["points"][0]["text"]
        imgpath = imgpath.replace(
            "/dkfz/cluster/gpu/data/OE0612/l049e/", "/home/t974t/Data/levin/"
        )

        label = hoverData["points"][0]["customdata"].split(",")[0]
        predicted = hoverData["points"][0]["customdata"].split(",")[1]

        with open(imgpath, "rb") as img:
            data = base64.b64encode(img.read()).replace(b"\n", b"").decode("utf-8")
            return [
                html.Img(
                    id="curimg",
                    width="512px",
                    height="512px",
                    src=f"data:image/jpeg;base64,{data}",
                ),
                html.H5(f"Label: {label}, Predicted: {predicted}, Path: {imgpath}"),
            ]

    @app.callback(
        Output("sidebar", "children"),
        Input("base-path-dd", "value"),
    )
    def update_path(value):
        app_state.path = app_state.base_path / value
        app_state.analyser = Analyser(path=app_state.base_path / value)
        children = (
            _sidebar_folder_selection(app_state)
            + _sidebar_class_selection(app_state)
            + _sidebar_dataset_selection(app_state)
            + _sidebar_color_selection(app_state)
        )
        return children

    @app.callback(
        Output("latentspace", "figure", allow_duplicate=True),
        State("latentspace", "figure"),
        Input("marker-size", "value"),
        Input("marker-alpha", "value"),
        prevent_initial_call=True,
    )
    def update_marker_size(figure, marker_size, marker_alpha):
        n_traces = len(figure["data"])
        patch = Patch()

        for i in range(n_traces):
            patch["data"][i]["marker"]["size"] = marker_size
            patch["data"][i]["opacity"] = marker_alpha

        return patch

    @app.callback(
        Output({"type": "failure-img", "id": ALL, "testset": ALL}, "style"),
        Input({"type": "failure-img", "id": ALL, "testset": ALL}, "n_clicks"),
    )
    def on_click_failure(n_clicks):
        if ctx.triggered_id is None:
            return [{"border": "5px solid var(--color-bg)"} for _ in ctx.outputs_list]

        output = []
        for o in ctx.outputs_list:
            if o["id"]["id"] == ctx.triggered_id["id"]:
                output.append({"border": "5px solid red"})
            else:
                output.append({"border": "5px solid var(--color-bg)"})
        return output

    @app.callback(
        Output("latentspace", "figure", allow_duplicate=True),
        State("latentspace", "figure"),
        Input({"type": "failure-img", "id": ALL, "testset": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_click_failure2(figure, n_clicks):
        if ctx.triggered_id is None:
            raise PreventUpdate

        if figure is None:
            raise PreventUpdate

        stats = app_state.analyser.get_coords_from_filename(
            ctx.triggered_id["id"], ctx.triggered_id["testset"]
        )
        print(f"{stats=}")
        print(f"{len(figure['data'])=}")
        patched_figure = Patch()
        for i in range(len(figure["data"])):
            if figure["data"][i]["name"] == "failure":
                del patched_figure["data"][i]

        patched_figure["data"].append(
            go.Scatter3d(
                x=[stats["0"]],
                y=[stats["1"]],
                z=[stats["2"]],
                opacity=0.4,
                mode="markers",
                name="failure",
                hoverinfo="name",
                marker={"size": 10, "color": "black", "symbol": "x"},
            )
        )
        return patched_figure

    app.run(host="0.0.0.0", debug=True, port="8055")


if __name__ == "__main__":
    main()
