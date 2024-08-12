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

from typing import TYPE_CHECKING, Optional, Union

import altair as alt
import gradio
import gradio as gr
import pandas as pd
import umap

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class ExplorerInterface(gradio.Interface):
    """
    https://altair-viz.github.io/gallery/scatter_linked_table.html
    https://altair-viz.github.io/gallery/scatter_href.html
    https://altair-viz.github.io/user_guide/interactions.html#selection-targets
    """

    def __init__(self, fn, inputs, outputs, **kwargs):
        super().__init__(fn, inputs, outputs, **kwargs)

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
        # Extract texts
        texts = dataframe[text_column].tolist()

        # Apply embedding reduction
        component_columns: list[str] = ["x", "y"]
        if all([col in dataframe.columns for col in component_columns]):
            umap_df = dataframe[component_columns]
        else:
            cls._set_embedding_model(cls, embedding_model)
            embeddings = cls.embedding_model.encode(texts, convert_to_numpy=True)
            reducer = umap.UMAP(n_components=2, **umap_kwargs)
            umap_embeddings = reducer.fit_transform(embeddings)
            # Create a DataFrame for plotting
            umap_df = pd.DataFrame(
                umap_embeddings,
                columns=component_columns,
            )

        # Include additional columns
        umap_df["Index"] = dataframe.index
        for col in dataframe.columns:
            umap_df[col] = dataframe[col]
        if score_column and score_column in dataframe.columns:
            umap_df["Size"] = umap_df[score_column].div(umap_df[score_column].mean())

        # Scatter Plot with color change on hover
        brush = alt.selection_interval()
        if label_column and label_column in dataframe.columns:
            umap_df[label_column] = dataframe[label_column]
            color = alt.condition(
                brush,
                alt.Color(f"{label_column}:N", scale=alt.Scale(scheme="category10")),
                alt.value("grey"),
            )
        else:
            color = alt.condition(brush, alt.value("steelblue"), alt.value("grey"))
        if score_column and score_column in dataframe.columns:
            size = alt.Size(
                "Size:Q",
                scale=alt.Scale(domain=[umap_df["Size"].min(), umap_df["Size"].max()]),
            )
            encode_kwargs = {"size": size}
        else:
            encode_kwargs = {}

        size_kwargs = {"width": 600, "height": 600}
        size_kwargs = {}
        points = (
            alt.Chart(umap_df, **size_kwargs)
            .mark_circle(size=60)
            .encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q"),
                tooltip=[
                    f"{col}:N"
                    for col in dataframe.columns
                    if col not in component_columns + ["Index" + "Size"]
                ],
                color=color,
                **encode_kwargs,
            )
            .add_params(brush)
        )

        # Data Tables
        ranked_text = (
            alt.Chart(umap_df, **size_kwargs)
            .mark_text(align="left")
            .encode(y=alt.Y("row_number:O").axis(None))
            .transform_filter(brush)
            .transform_window(row_number="row_number()")
            .transform_filter(alt.datum.row_number < 25)
        )
        text_charts = []
        for col in dataframe.columns:
            if col in component_columns + ["Index" + "Size"]:
                continue
            umap_df[f"{col}_short"] = umap_df[col].str.slice(0, 100)
            text_charts.append(
                ranked_text.encode(text=f"{col}_short:N").properties(
                    title=alt.Title(text=col, align="left")
                )
            )
        text = alt.hconcat(*text_charts)

        # Build chart
        plot = (
            alt.hconcat(points, text)
            .resolve_legend(color="independent", size="independent")
            .configure_view(stroke=None)
            .configure_legend(orient="left")
        )

        # Gradio inputs and outputs
        gr_plot = gr.Plot(plot)
        inputs = [gr_plot]

        # Logic update function
        def _test(_gr_plot):
            return _gr_plot

        return cls(fn=_test, inputs=inputs, outputs=inputs)
