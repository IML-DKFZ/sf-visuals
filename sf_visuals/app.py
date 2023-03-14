import argparse
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, dcc, html

from sf_visuals.analyser import Analyser


def main():
    app = Dash(__name__)
    colors = {"background": "white", "text": "black"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    analyser = Analyser(path=args.path)
    analyser.setup()

    app.layout = html.Div(
        [
            html.Div(
                id="sidebar",
                style={"display": "inline-block"},
                children=[
                    html.H2(
                        children=f"Experiment folder:",
                    ),
                    html.P(f"{args.path}"),
                    html.H2(
                        children="Select datafile and class to display:",
                    ),
                    dcc.Checklist(
                        [
                            {
                                "label": html.Div(
                                    [
                                        f"Class {c}: ",
                                        dcc.Input(
                                            id={"type": "class-name", "id": f"{c}"},
                                            type="text",
                                            value=f"{c}",
                                        ),
                                    ],
                                    style={"display": "inline-block"},
                                ),
                                "value": c,
                            }
                            for c in analyser.classes
                        ],
                        analyser.classes,
                        id="checklist-classes",
                    ),
                    html.Div(id="debug", children=[]),
                    dcc.RadioItems(
                        [
                            {
                                "label": html.Div(
                                    [
                                        f"Testset {c}: ",
                                        dcc.Input(
                                            id={"type": "testset-name", "id": f"{c}"},
                                            type="text",
                                            value=f"{c}",
                                        ),
                                    ],
                                    style={"display": "inline-block"},
                                ),
                                "value": c,
                            }
                            for c in analyser.testsets
                        ],
                        analyser.testsets[0],
                        id="checklist-testsets",
                    ),
                ],
            ),
            dcc.Graph(
                figure=analyser._Analyser__encoderls[1][0],
                style={"display": "inline-block", "width": "80%"},
            ),
        ],
        style={"display": "flex"}
    )

    @app.callback(
        Output("debug", "children"),
        Input({"type": "class-name", "id": ALL}, "value"),
        State({"type": "class-name", "id": ALL}, "id"),
    )
    def update_class_name(value, id):
        classes = {}
        for i, v in zip(id, value):
            classes[int(i["id"])] = v
        analyser.classes = classes
        return [str(analyser._Analyser__class2name)]

    # @app.callback(
    #     Output("scatter", "figure"),
    #     Input("dropdown_class", "value"),
    #     Input("dropdown_data", "value"),
    # )
    # def update_encoder(dropdown_class, dropdown_data):
    #     global TEXT_ARRAY
    #     TEXT_ARRAY = TEXT_ARRAYS[dropdown_data][dropdown_class]
    #     figure = FIGURES[dropdown_data][dropdown_class]
    #
    #     return figure
    #
    #
    # @app.callback(Output("dropdown_class", "options"), Input("dropdown_data", "value"))
    # def update_class_dropdown(data_name):
    #     df = df_dict[data_name]
    #     return np.sort(df.label.unique())
    #
    #
    # @app.callback(
    #     Output("img_plot", "figure"),
    #     # Output("test", "children"),
    #     Input("scatter", "hoverData"),
    # )
    # def update_on_hover(hoverData):
    #     """ """
    #     if hoverData is None:
    #         raise PreventUpdate
    #     try:
    #         imgpath = hoverData["points"][0]["text"]
    #         _, end = imgpath.split("/l049e")
    #         if "cluster" in imgpath:
    #             filep2 = os.path.join("/home/l049e/Data/" + end)
    #         if "home" in imgpath:
    #             filep2 = imgpath
    #         imbgr2 = cv2.imread(filep2)
    #         im2 = cv2.cvtColor(imbgr2, cv2.COLOR_BGR2RGB)
    #         fig2 = go.Figure()
    #         fig2.add_trace(go.Image(z=im2))
    #         fig2.update_layout(width=1000, height=1000, template="simple_white")
    #         return fig2
    #         # return filep2
    #     except Exception as error:
    #         print(error)
    #         raise PreventUpdate
    app.run(host="0.0.0.0", debug=True, port="8055")


if __name__ == "__main__":
    main()
