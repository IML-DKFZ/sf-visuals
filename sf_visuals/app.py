import argparse
import base64
from dataclasses import dataclass
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from sf_visuals.analyser import Analyser


@dataclass
class AppState:
    path: Path
    analyser: Analyser


def main():
    app = Dash(__name__)
    colors = {"background": "white", "text": "black"}

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
            dcc.RadioItems(
                [{"label": "All", "value": "ALL"}]
                + [
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
                ],
                "ALL",
                id="checklist-testsets",
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

    app.layout = dcc.Tabs(
        parent_className="container",
        className="tab-container",
        children=[
            dcc.Tab(
                label="Latent Space",
                children=html.Div(
                    id="tab-latentspace",
                    className="tab-custom",
                    children=[
                        sidebar,
                        dcc.Graph(
                            id="latentspace",
                            className="latentspace",
                            responsive=True,
                            clear_on_unhover=True,
                        ),
                        html.Div(
                            className="hover-preview",
                            children=[
                                html.Img(id="curimg", width="512px", height="512px"),
                            ],
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
                    children=[html.H2("WIP", className="failure-view")],
                ),
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
        Input("checklist-classes", "value"),
        Input("checklist-colorby", "value"),
    )
    def update_testset(testset, classes, colorby):
        figure =  app_state.analyser.plot_latentspace(
            testset, classes2plot=tuple(classes), coloring=colorby
        )

        figure["layout"]["uirevision"] = True

        return figure

    @app.callback(
        Output("curimg", "src"),
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
            return f"data:image/jpeg;base64,{data}"

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
