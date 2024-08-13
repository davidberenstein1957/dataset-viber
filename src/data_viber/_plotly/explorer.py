# Copyright 2024-present, David Berenstein, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Optional, Union

import dash
import pandas as pd
import plotly.express as px
import umap
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from data_viber._constants import DEFAULT_EMBEDDING_MODEL
from plotly.graph_objs._figure import Figure

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


class ExplorerInterface:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        *,
        label_column: str = None,
        score_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = DEFAULT_EMBEDDING_MODEL,
        umap_kwargs: dict = {},
        labels: list[str] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ):
        self.text_column = text_column
        self.label_column = label_column
        self.labels = labels

        # Extract texts
        texts = dataframe[text_column].tolist()

        # Apply embedding reduction
        component_columns: list[str] = ["x", "y"]
        if all([col in dataframe.columns for col in component_columns]):
            umap_df = dataframe[component_columns]
        else:
            self._set_embedding_model(embedding_model)
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            reducer = umap.UMAP(n_components=2, **umap_kwargs)
            umap_embeddings = reducer.fit_transform(embeddings)
            # Create a DataFrame for plotting
            umap_df = pd.DataFrame(
                umap_embeddings,
                columns=component_columns,
            )

        umap_df["index"] = dataframe.index
        for col in dataframe.columns:
            umap_df[col] = dataframe[col]
        if score_column and score_column in dataframe.columns:
            umap_df["Size"] = umap_df[score_column].div(umap_df[score_column].mean())

        self.umap_df = umap_df
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        figure = self._get_initial_figure(umap_df)
        app.layout = self._get_app_layout(figure, umap_df, labels, hf_token)

        if labels is not None:

            @app.callback(
                [
                    Output("scatter-plot", "figure", allow_duplicate=True),
                    Output("data-table", "data", allow_duplicate=True),
                ],
                [Input("update-button", "n_clicks")],
                [
                    State("scatter-plot", "selectedData"),
                    State("label-dropdown", "value"),
                    State("scatter-plot", "figure"),
                ],
                prevent_initial_call=True,
            )
            def update_labels(n_clicks, selectedData, new_label, current_figure):
                ctx = dash.callback_context
                if not ctx.triggered or new_label is None:
                    return current_figure, self.umap_df.to_dict("records")
                if selectedData and selectedData["points"]:
                    selected_indices = [
                        point["customdata"][0] for point in selectedData["points"]
                    ]
                    self.umap_df.loc[selected_indices, self.label_column] = new_label
                    print(self.umap_df)
                    updated_traces = []
                    points_to_move = defaultdict(list)
                    for trace in current_figure["data"]:
                        if new_label != trace["name"]:
                            points_to_keep = defaultdict(list)
                            for idx, point in enumerate(trace["customdata"]):
                                if point[0] not in selected_indices:
                                    points_to_keep["customdata"].append(point)
                                    points_to_keep["x"].append(trace["x"][idx])
                                    points_to_keep["y"].append(trace["y"][idx])
                                else:
                                    points_to_move["customdata"].append(point)
                                    points_to_move["x"].append(trace["x"][idx])
                                    points_to_move["y"].append(trace["y"][idx])
                            trace["customdata"] = points_to_keep["customdata"]
                            trace["x"] = points_to_keep["x"]
                            trace["y"] = points_to_keep["y"]
                            trace["selectedpoints"] = []
                            updated_traces.append(trace)
                    for trace in current_figure["data"]:
                        if trace["name"] == new_label:
                            trace["customdata"] += points_to_move["customdata"]
                            trace["x"] += points_to_move["x"]
                            trace["y"] += points_to_move["y"]
                            trace["selectedpoints"] = []
                            updated_traces.append(trace)
                    current_figure["data"] = updated_traces
                    print(self.umap_df)
                    return current_figure, self.umap_df.to_dict("records")

                return current_figure, self.umap_df.to_dict("records")

            # Callback to print the dataframe
            @app.callback(
                Output("data-table", "data", allow_duplicate=True),
                [Input("upload-button", "n_clicks")],
                prevent_initial_call=True,
            )
            def print_dataframe(n_clicks):
                if n_clicks > 0:
                    print(self.umap_df)  # This will print the dataframe to the console
                return self.umap_df.to_dict(
                    "records"
                )  # Return the data to avoid updating the table

            @app.callback(
                Output("download-text", "data"),
                Input("btn-download-txt", "n_clicks"),
                prevent_initial_call=True,
            )
            def func(n_clicks):
                return dcc.send_data_frame(self.umap_df.to_csv, "data.csv")

        # Update the existing update_selection callback
        @app.callback(
            [
                Output("scatter-plot", "figure", allow_duplicate=True),
                Output("data-table", "data", allow_duplicate=True),
            ],
            [Input("scatter-plot", "selectedData")],
            [State("scatter-plot", "figure")],
            prevent_initial_call=True,
        )
        def update_selection(selectedData, figure):
            ctx = dash.callback_context
            if not ctx.triggered:
                return figure, self.umap_df.to_dict("records")

            if selectedData and selectedData["points"]:
                selected_indices = [p["pointIndex"] for p in selectedData["points"]]
                filtered_df = self.umap_df.iloc[selected_indices]
            else:
                filtered_df = self.umap_df
                selected_indices = None

            return figure, filtered_df.to_dict("records")

        self.app = app

    def _set_embedding_model(self, embedding_model: str):
        import torch

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            if importlib.util.find_spec("onnxruntime") is not None:
                from data_viber.embedder import Embedder

                self.embedding_model = Embedder(model_id=embedding_model, device=device)
            else:
                self.embedding_model = SentenceTransformer(
                    model_name_or_path=embedding_model, device=device
                )
        else:
            raise ValueError(
                "Embedding model should be of type `str` or `SentenceTransformer`"
            )

    @classmethod
    def for_text_visualization(
        cls,
        dataframe: pd.DataFrame,
        text_column: str,
        *,
        label_column: str = None,
        score_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = "all-MiniLM-L6-v2",
        umap_kwargs: dict = {},
    ):
        return cls(
            dataframe,
            text_column,
            label_column=label_column,
            score_column=score_column,
            embedding_model=embedding_model,
            umap_kwargs=umap_kwargs,
        )

    @classmethod
    def for_text_classification(
        cls,
        dataframe: pd.DataFrame,
        text_column: str,
        labels: list[str],
        *,
        label_column: str = None,
        score_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = "all-MiniLM-L6-v2",
        umap_kwargs: dict = {},
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ):
        if not label_column:
            dataframe["label"] = ""
            label_column = "label"
        return cls(
            dataframe,
            text_column,
            label_column=label_column,
            score_column=score_column,
            embedding_model=embedding_model,
            umap_kwargs=umap_kwargs,
            labels=labels,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    def launch(self, **kwargs):
        self.app.run_server(**kwargs)

    def _get_initial_figure(self, dataframe) -> Figure:
        # color_map = {label: color for label, color in zip(self.labels, _COLORS)}
        dataframe[f"wrapped_hover_{self.text_column}"] = dataframe[
            self.text_column
        ].apply(lambda x: self.wrap_text(x, max_length=30))
        custom_data = ["index"] + [
            col
            for col in dataframe.columns
            if col not in ["x", "y", "index", self.text_column]
        ]
        hovertemplate: Literal[""] = ""
        df_custom = dataframe[custom_data]
        for col in df_custom.columns:
            if col in ["index", self.label_column]:
                continue
            idx = df_custom.columns.get_loc(col)
            hovertemplate += f"<b>{col.replace('wrapped_hover_', '')}</b>:<br>%{{customdata[{idx}]}}<br>"
        fig = px.scatter(
            dataframe,
            x="x",
            y="y",
            color=self.label_column if self.label_column in dataframe.columns else None,
            height=800,
            custom_data=custom_data,
        )
        print(hovertemplate)
        fig.update_traces(hovertemplate=str(hovertemplate))

        fig.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode="lasso",
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="top",  # Anchor the legend to the top of the container
                y=-0.01,  # Position the legend below the plot
                xanchor="center",  # Center the legend horizontally
                x=0.5,  # Center the legend at the bottom
            ),
        )
        return fig

    def _get_app_layout(self, figure, dataframe, labels, hf_token):
        buttons = []
        if labels is not None:
            buttons.extend(
                [
                    buttons.append(
                        dcc.Dropdown(
                            id="label-dropdown",
                            options=[
                                {"label": label, "value": label} for label in labels
                            ],
                            value=labels[0],
                            clearable=True,
                            style={
                                "width": "200px",
                                "marginBottom": "-13px",
                                "display": "inline-block",
                            },
                        )
                    ),
                    buttons.append(
                        html.Button(
                            "Update Labels",
                            id="update-button",
                            n_clicks=0,
                        )
                    ),
                    html.Button(
                        "Upload to Hub",
                        id="upload-button",
                        n_clicks=0,
                    ),
                    html.Button("Download Text", id="btn-download-txt"),
                    dcc.Download(id="download-text"),
                ]
            )
        layout = html.Div(
            [
                html.H1("ExplorerInterface"),
                # Scatter plot
                html.Div(
                    [
                        dcc.Graph(id="scatter-plot", figure=figure),
                        html.Div(
                            [
                                *buttons,
                                # dcc.Upload(
                                #     id='upload-data',
                                #     children=html.Div([
                                #         'Drag and Drop or ',
                                #         html.A('Select Files')
                                #     ]),
                                #     style={
                                #         'width': '200px',
                                #         # 'height': '60px',
                                #         # 'lineHeight': '60px',
                                #         'borderWidth': '1px',
                                #         'borderStyle': 'dashed',
                                #         'borderRadius': '5px',
                                #         'textAlign': 'center',
                                #         "display": "inline-block"
                                #         # 'margin': '10px'
                                #     },
                                #     # Allow multiple files to be uploaded
                                #     multiple=False
                                # ),
                                # html.Div(id='output-data-upload'),
                            ]
                        ),
                    ],
                    style={
                        "width": "49%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "marginRight": "1%",
                    },
                ),
                html.Div(
                    [
                        dash_table.DataTable(
                            id="data-table",
                            columns=[{"name": i, "id": i} for i in dataframe.columns],
                            data=dataframe.to_dict("records"),
                            hidden_columns=[
                                "x",
                                "y",
                                "index",
                                f"wrapped_hover_{self.text_column}",
                            ],
                            column_selectable=False,
                            page_size=20,
                            fill_width=True,
                            css=[{"selector": ".show-hide", "rule": "display: none"}],
                            style_cell={
                                "whiteSpace": "normal",
                                "height": "auto",
                                "textAlign": "left",
                            },
                            style_data={
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_table={"overflowX": "auto"},
                        )
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                ),
            ]
        )
        return layout

    @staticmethod
    def wrap_text(text, max_length=60):
        words = text.split(" ")
        wrapped_text = ""
        line = ""

        for word in words:
            if len(line) + len(word) + 1 > max_length:
                if line:
                    wrapped_text += line + "<br>"
                line = word
            else:
                if line:
                    line += " " + word
                else:
                    line = word

        wrapped_text += line
        return wrapped_text
