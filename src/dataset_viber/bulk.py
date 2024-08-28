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
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import umap
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from plotly.graph_objs._figure import Figure

from dataset_viber._constants import DEFAULT_EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class BulkInterface:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        content_column: str,
        *,
        label_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = DEFAULT_EMBEDDING_MODEL,
        umap_kwargs: dict = {},
        labels: list[str] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
        content_format: Optional[str] = "text",
    ):
        self.content_format = content_format
        self.content_column = content_column
        self.label_column = label_column
        self.labels = labels
        if label_column and labels:
            if not all([label in dataframe[label_column].unique() for label in labels]):
                # apply label to first x empty rows in label_column
                empty_rows = dataframe[dataframe[label_column] == ""].index
                for label, row in zip(labels, empty_rows):
                    dataframe.loc[row, label_column] = label
                warnings.warn(
                    "Labels were not found in the label_column. Applied labels to the first x empty rows."
                )

        contents = dataframe[content_column].tolist()

        # Apply embedding reduction
        component_columns: list[str] = ["x", "y"]
        if all([col in dataframe.columns for col in component_columns]):
            umap_df = dataframe[component_columns]
        else:
            if "embeddings" in dataframe.columns:
                embeddings = dataframe["embeddings"].tolist()
            else:
                self._set_embedding_model(embedding_model)
                embeddings = self.embed_content(contents)
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

        self.umap_df = umap_df
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
        figure = self._get_initial_figure(umap_df)
        app.layout = self._get_app_layout(figure, umap_df, labels, hf_token)

        if labels is not None:

            @app.callback(
                [
                    Output("scatter-plot", "figure", allow_duplicate=True),
                    Output("data-table", "data", allow_duplicate=True),
                    Output("data-table", "tooltip_data", allow_duplicate=True),
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

                hidden_traces = []
                for trace in figure["data"]:
                    if trace["visible"] == "legendonly":
                        hidden_traces.append(trace)

                if selectedData and selectedData["points"]:
                    selected_indices = [
                        point["customdata"][0] for point in selectedData["points"]
                    ]
                    self.umap_df.loc[selected_indices, self.label_column] = new_label
                    updated_traces = []
                    points_to_move = defaultdict(list)
                    for trace in current_figure["data"]:
                        if trace not in hidden_traces:
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

                local_dataframe = self.umap_df.copy()
                tooltip_data = self.get_tooltip(local_dataframe)
                if self.content_format == "chat":
                    local_dataframe[self.content_column] = local_dataframe[
                        self.content_column
                    ].apply(lambda x: x[0]["content"])
                return current_figure, local_dataframe.to_dict("records"), tooltip_data

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
                Output("data-table", "tooltip_data", allow_duplicate=True),
            ],
            [Input("scatter-plot", "selectedData")],
            [State("scatter-plot", "figure")],
            prevent_initial_call=True,
        )
        def update_selection(selectedData, figure):
            ctx = dash.callback_context
            if not ctx.triggered:
                return figure, self.umap_df.to_dict("records")

            hidden_traces = []
            for trace in figure["data"]:
                if trace.get("visible") == "legendonly":
                    hidden_traces.append(trace)

            if selectedData and selectedData["points"]:
                selected_indices = [
                    point["customdata"][0] for point in selectedData["points"]
                ]
                filtered_df = self.umap_df.iloc[selected_indices]
            else:
                filtered_df = self.umap_df
                selected_indices = None

            if hidden_traces:
                filtered_df = filtered_df[
                    ~filtered_df[self.label_column].isin(hidden_traces)
                ]

            local_dataframe = filtered_df.copy()
            tooltip_data = self.get_tooltip(local_dataframe)
            if self.content_format == "chat":
                local_dataframe[self.content_column] = local_dataframe[
                    self.content_column
                ].apply(lambda x: x[0]["content"])
            return figure, local_dataframe.to_dict("records"), tooltip_data

        self.app = app

    def _set_embedding_model(self, embedding_model: str):
        import torch
        from sentence_transformers import SentenceTransformer

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            if importlib.util.find_spec("fast_sentence_transformers") is not None:
                from fast_sentence_transformers import FastSentenceTransformer

                self.embedding_model = FastSentenceTransformer(
                    model_id=embedding_model, device=device
                )
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
        content_column: str,
        *,
        label_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = DEFAULT_EMBEDDING_MODEL,
        umap_kwargs: dict = {},
    ):
        return cls(
            dataframe=dataframe,
            content_column=content_column,
            label_column=label_column,
            embedding_model=embedding_model,
            umap_kwargs=umap_kwargs,
            content_format="text",
        )

    @classmethod
    def for_text_classification(
        cls,
        dataframe: pd.DataFrame,
        content_column: str,
        labels: list[str],
        *,
        label_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = DEFAULT_EMBEDDING_MODEL,
        umap_kwargs: dict = {},
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ):
        if not label_column:
            dataframe["label"] = ""
            label_column = "label"
        return cls(
            dataframe=dataframe,
            content_column=content_column,
            label_column=label_column,
            embedding_model=embedding_model,
            umap_kwargs=umap_kwargs,
            labels=labels,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
            content_format="text",
        )

    @classmethod
    def for_chat_visualization(
        cls,
        dataframe: pd.DataFrame,
        chat_column: List[Dict[str, str]],
        *,
        label_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = DEFAULT_EMBEDDING_MODEL,
        umap_kwargs: dict = {},
    ):
        return cls(
            dataframe=dataframe,
            content_column=chat_column,
            label_column=label_column,
            embedding_model=embedding_model,
            umap_kwargs=umap_kwargs,
            content_format="chat",
        )

    @classmethod
    def for_chat_classification(
        cls,
        dataframe: pd.DataFrame,
        chat_column: List[Dict[str, str]],
        labels: list[str],
        *,
        label_column: str = None,
        embedding_model: Optional[
            Union["SentenceTransformer", str]
        ] = DEFAULT_EMBEDDING_MODEL,
        umap_kwargs: dict = {},
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ):
        if not label_column:
            dataframe["label"] = ""
            label_column = "label"
        return cls(
            dataframe=dataframe,
            content_column=chat_column,
            embedding_model=embedding_model,
            umap_kwargs=umap_kwargs,
            labels=labels,
            dataset_name=dataset_name,
            label_column=label_column,
            hf_token=hf_token,
            private=private,
            content_format="chat",
        )

    def launch(self, **kwargs):
        self.app.run_server(**kwargs)

    def _get_initial_figure(self, dataframe) -> Figure:
        # color_map = {label: color for label, color in zip(self.labels, _COLORS)}
        dataframe[f"wrapped_hover_{self.content_column}"] = dataframe[
            self.content_column
        ].apply(lambda x: self.format_content(x, content_format=self.content_format))
        custom_data = ["index"] + [
            col
            for col in dataframe.columns
            if col not in ["x", "y", "index", self.content_column]
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
            hoverlabel=dict(
                font_size=10,
                font_family="monospace",
            ),
        )
        return fig

    def _get_app_layout(self, figure, dataframe, labels, hf_token):
        local_dataframe = dataframe.copy()
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
                        dbc.Button(
                            "Update Labels",
                            id="update-button",
                            n_clicks=0,
                        )
                    ),
                    dbc.Button(
                        "Upload to Hub",
                        id="upload-button",
                        n_clicks=0,
                    ),
                    dbc.Button("Download Text", id="btn-download-txt"),
                    dcc.Download(id="download-text"),
                ]
            )
        if self.content_format == "chat":
            tooltip_data = self.get_tooltip(local_dataframe)
            local_dataframe[self.content_column] = local_dataframe[
                self.content_column
            ].apply(lambda x: x[0]["content"])
            columns = local_dataframe.columns
        elif self.content_format == "text":
            tooltip_data = None
            columns = local_dataframe.columns
        else:
            raise ValueError(
                "content_format should be either 'text' or 'chat' but got {self.content_format}"
            )

        layout = html.Div(
            [
                html.H1("BulkInterface"),
                # Scatter plot
                html.Div(
                    [
                        dcc.Graph(id="scatter-plot", figure=figure),
                        html.Div([*buttons]),
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
                            columns=[{"name": i, "id": i} for i in columns],
                            data=local_dataframe[columns].to_dict("records"),
                            hidden_columns=[
                                "x",
                                "y",
                                "index",
                                f"wrapped_hover_{self.content_column}",
                            ],
                            tooltip_data=tooltip_data,
                            column_selectable=False,
                            page_size=20,
                            fill_width=True,
                            css=[
                                {"selector": ".show-hide", "rule": "display: none"},
                                {
                                    "selector": ".dash-table-tooltip",
                                    "rule": """
                                        background-color: grey;
                                        font-family: monospace;
                                        color: white;
                                        max-width: 100vw !important;
                                        max-height: 80vh !important;
                                        overflow: auto;
                                        font-size: 10px;
                                        position: fixed;
                                        top: 50%;
                                        left: 50%;
                                        transform: translate(-50%, -50%);
                                        z-index: 1000;
                                    """,
                                },
                            ],
                            style_cell={
                                "whiteSpace": "normal",
                                "height": "auto",
                                "textAlign": "left",
                                "font-size": "10px",
                                "overflow": "auto",  # Enable scrolling
                            },
                            style_data={
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_table={"overflowX": "auto"},
                            tooltip_duration=None,
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

    def embed_content(self, content: List[str]):
        if self.content_format == "text":
            return self.embedding_model.encode(content, convert_to_numpy=True)
        elif self.content_format == "chat":
            content = [
                " ".join([turn["content"] for turn in conversation])
                for conversation in content
            ]
            return self.embedding_model.encode(content, convert_to_numpy=True)

    def format_content(self, content, max_length=120, content_format="text"):
        wrapped_text = ""
        if content_format == "text":
            words = content.split(" ")
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
        elif content_format == "chat":
            wrapped_text = "First 2 turns:<br><br>"
            for turn in content[:3]:
                wrapped_text += f"<b>{turn['role']}</b>:<br>{self.format_content(turn['content'])}<br><br>"
            return wrapped_text

    def get_tooltip(self, dataframe):
        if self.content_format == "text":
            return None
        return [
            {
                self.content_column: {
                    "value": pd.DataFrame.from_records(value)[
                        ["role", "content"]
                    ].to_markdown(index=False, tablefmt="pipe"),
                    "type": "markdown",
                }
            }
            for value in dataframe[self.content_column].tolist()
        ]
