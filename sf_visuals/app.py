from __future__ import annotations

import argparse
import base64
import itertools
from dataclasses import dataclass
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, dcc, html
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
                        dcc.Graph(
                            id="latentspace",
                            className="latentspace",
                            responsive=True,
                            clear_on_unhover=True,
                        ),
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


def _failure_triplet(testset: str, data: str, stats: list[dict]):
    return html.Div(
        children=[
            html.H5(f"Testset: {testset}"),
            html.Div(
                [
                    html.Img(
                        id="curimg",
                        className="failure-img",
                        # width="512px",
                        height="512px",
                        src=f"data:image/svg;base64,{data}",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P(f"Pr: {stats[0]['pred']}"),
                                    html.P(f"GT: {stats[0]['label']}"),
                                    html.P(f"C : {stats[0]['conf']}"),
                                ],
                                className="failure-stat",
                            ),
                            html.Div(
                                [
                                    html.P(f"Pr: {stats[1]['pred']}"),
                                    html.P(f"GT: {stats[1]['label']}"),
                                    html.P(f"C : {stats[1]['conf']}"),
                                ],
                                className="failure-stat",
                            ),
                            html.Div(
                                [
                                    html.P(f"Pr: {stats[2]['pred']}"),
                                    html.P(f"GT: {stats[2]['label']}"),
                                    html.P(f"C : {stats[2]['conf']}"),
                                ],
                                className="failure-stat",
                            ),
                        ],
                        className="failure-desc",
                    ),
                ],
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
            svg, stats = app_state.analyser.overconfident(testset)
            data = base64.b64encode(svg).replace(b"\n", b"").decode("utf-8")
            imgs.append(_failure_triplet(testset, data, stats))

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
        with open(imgpath, "rb") as img:
            data = base64.b64encode(img.read()).replace(b"\n", b"").decode("utf-8")
            return html.Img(
                id="curimg",
                width="512px",
                height="512px",
                src=f"data:image/jpeg;base64,{data}",
            )

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

    app.run(host="0.0.0.0", debug=True, port="8055")


if __name__ == "__main__":
    main()
