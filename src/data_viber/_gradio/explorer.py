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

import altair as alt
import gradio as gr
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer


class ExplorerInterface(gr.Interface):
    """
    https://altair-viz.github.io/gallery/scatter_linked_table.html
    https://altair-viz.github.io/gallery/scatter_href.html
    """

    def __init__(self, fn, inputs, outputs, **kwargs):
        super().__init__(fn, inputs, outputs, **kwargs)

    @classmethod
    def for_dataframe_visualization(
        cls,
        dataframe: pd.DataFrame,
        text_column: str,
        additional_columns: list = None,
        label_column: str = None,
        score_column: str = None,
        umap_n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if additional_columns is None:
            additional_columns = []

        # Extract texts
        texts = dataframe[text_column].tolist()

        # Embed the texts using sentence-transformers
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Apply UMAP for dimensionality reduction
        reducer = umap.UMAP(
            n_components=umap_n_components, n_neighbors=n_neighbors, min_dist=min_dist
        )
        umap_embeddings = reducer.fit_transform(embeddings)

        # Create a DataFrame for plotting
        umap_df = pd.DataFrame(
            umap_embeddings,
            columns=[f"Component_{i+1}" for i in range(umap_n_components)],
        )
        umap_df[text_column] = texts
        umap_df["Index"] = dataframe.index

        # Include additional columns
        for col in additional_columns:
            if col in dataframe.columns:
                umap_df[col] = dataframe[col]

        if label_column and label_column in dataframe.columns:
            umap_df[label_column] = dataframe[label_column]

        if score_column and score_column in dataframe.columns:
            umap_df[score_column] = dataframe[score_column]
            umap_df["Size"] = umap_df[score_column].div(umap_df[score_column].mean())

        # Brush for selection
        brush = alt.selection_interval()

        # Scatter Plot with color change on hover
        points = (
            alt.Chart(umap_df)
            .mark_point()
            .encode(
                x=alt.X("Component_1:Q"),
                y=alt.Y("Component_2:Q"),
                color=alt.condition(
                    brush,
                    alt.Color(
                        f"{label_column}:N", scale=alt.Scale(scheme="category10")
                    ),
                    alt.value("grey"),
                ),
                size=alt.Size(
                    "Size:Q",
                    scale=alt.Scale(
                        domain=[umap_df["Size"].min(), umap_df["Size"].max()]
                    ),
                )
                if score_column
                else None,
                tooltip=[text_column + ":N"]
                + [f"{col}:N" for col in additional_columns],
            )
            .add_params(brush)
        )

        # Data Tables
        ranked_text = (
            alt.Chart(umap_df)
            .mark_text(align="left")
            .encode(y=alt.Y("row_number:O").axis(None))
            .transform_filter(brush)
            .transform_window(row_number="row_number()")
            .transform_filter(alt.datum.row_number < 15)
        )

        text_charts = [
            ranked_text.encode(text=text_column + ":N").properties(
                title=alt.Title(text=text_column, align="left")
            )
        ]

        for col in additional_columns:
            text_charts.append(
                ranked_text.encode(text=f"{col}:N").properties(
                    title=alt.Title(text=col, align="left")
                )
            )

        # Combine data tables horizontally
        text = alt.hconcat(*text_charts)

        # Build chart
        plot = (
            alt.hconcat(points, text)
            .resolve_legend(color="independent", size="independent")
            .configure_view(stroke=None)
            .configure_legend(orient="left")
        )

        # Gradio inputs and outputs
        inputs = gr.Plot(plot)

        def _test(x):
            print(x)
            return x

        # Return an instance of the class
        return cls(fn=_test, inputs=inputs, outputs=inputs)


# Example usage:
if __name__ == "__main__":
    # Example DataFrame with text, label, and score columns
    df = pd.DataFrame(
        {
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "The quick brown fox jumps over the lazy dog.",
                "A journey of a thousand miles begins with a single step.",
                "To be or not to be, that is the question.",
            ],
            "category": ["A", "A", "B", "C"],
            "length": [44, 44, 36, 35],
            "score": [10, 20, 30, 40],  # This column will size the points
        }
    )

    visualizer = ExplorerInterface.for_dataframe_visualization(
        df,
        text_column="text",
        additional_columns=["category", "length"],
        label_column="category",
        score_column="score",
    )
    visualizer.launch()
