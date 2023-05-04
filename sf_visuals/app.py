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
    path: Path
    analyser: Analyser


def main():
    app = Dash(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    app_state = AppState(path=args.path, analyser=Analyser(path=args.path))
    base_path = Path(
        "/home/t974t/NetworkDrives/E130-Personal/Kobelke/cluster_checkpoints/"
    )
    all_paths = sorted(
        [
            str(path.parent.relative_to(base_path))
            for path in base_path.glob("**/test_results")
        ]
    )

    def sidebar_content(app_state: AppState):
        return [
            html.H2(
                children="Experiment folder:",
            ),
            dcc.Dropdown(
                all_paths,
                str(app_state.path.relative_to(base_path)),
                id="base-path-dd",
                clearable=False,
            ),
            html.Div(id="path-display-dd"),
            html.H3(
                children="Classes:",
            ),
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
            html.H3(
                children="Datasets:",
            ),
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
                                [
                                    f"Testset {c}",
                                ],
                                style={"display": "inline-block"},
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
            html.H3(
                children="Color By:",
            ),
            dcc.RadioItems(
                [
                    {"label": "Confidence", "value": "confidence"},
                    {
                        "label": "Source/Target",
                        "value": "source-target",
                    },
                ],
                "confidence",
                id="checklist-colorby",
            ),
        ]

    sidebar = html.Div(
        id="sidebar",
        style={"display": "inline-block"},
        children=sidebar_content(app_state),
    )

    app.layout = html.Div(
        className="app-container",
        children=[
            sidebar,
            dcc.Tabs(
                parent_className="container",
                className="tab-container",
                children=[
                    dcc.Tab(
                        label="Latent Space",
                        children=html.Div(
                            id="tab-latentspace",
                            className="tab-custom",
                            children=[
                                dcc.Loading(
                                    dcc.Graph(
                                        id="latentspace",
                                        className="latentspace",
                                        responsive=True,
                                        clear_on_unhover=True,
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
                    ),
                    dcc.Tab(
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
                                    # className="latentspace-loading",
                                ),
                                dcc.Loading(
                                    html.Div(
                                        id="failure-view",
                                        className="failure-view",
                                    ),
                                ),
                            ],
                        ),
                    ),
                ],
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
        return None

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
            imgs.append(
                html.Div(
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
            )

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
        app_state.path = base_path / value
        app_state.analyser = Analyser(path=base_path / value)
        return sidebar_content(app_state)

    app.run(host="0.0.0.0", debug=True, port="8055")


if __name__ == "__main__":
    main()
