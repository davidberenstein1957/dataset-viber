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

import io
import json
import threading
import time
import uuid
from pathlib import Path

import gradio
import numpy as np
import pandas as pd
from dataset_viber._gradio._mixins._argilla import ArgillaMixin
from datasets import Dataset, load_dataset
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from PIL import Image

CODE_KWARGS = {
    "language": "json",
    "interactive": True,
    "label": "Column Mapping",
    "lines": 1,
}


class ImportExportMixin(ArgillaMixin):
    def _override_block_init_method(self, **kwargs):
        # Initialize the parent class
        gradio.Blocks.__init__(
            self,
            analytics_enabled=kwargs.get("analytics_enabled", True),
            mode="interface",
            css=kwargs.get("css", None),
            title=kwargs.get("title", "Gradio"),
            theme=kwargs.get("theme", None),
            js=kwargs.get("js", None),
            head=kwargs.get("head", None),
            delete_cache=kwargs.get("delete_cache", False),
            fill_width=kwargs.get("fill_width", False),
            # **kwargs,
        )
        # Override the __init__ method of the parent class to avoid the re-creation of the blocks
        gradio.Blocks.__init__ = lambda *args, **kwargs: None

    def _configure_import(self):
        with gradio.Tab("Import data"):
            with gradio.Tab("Import from Hugging Face Hub"):
                search_in = HuggingfaceHubSearch(
                    label="Search Huggingface Hub",
                    placeholder="Search for datasets on Huggingface",
                    search_type="dataset",
                    sumbit_on_select=True,
                )
                dataset_viewer = gradio.HTML(label="Dataset Viewer")
                search_in.submit(
                    fn=lambda x: self._get_embedded_dataset_viewer(
                        self._get_repo_url_from_repo_id(x)
                    ),
                    inputs=[search_in],
                    outputs=[dataset_viewer],
                )
                column_mapping_hf_upload = gradio.Code(
                    value=json.dumps(dict.fromkeys(self.input_columns, ""), indent=2),
                    **CODE_KWARGS,
                )
                start_btn_hf_upload = gradio.Button("Start Annotating")
                start_btn_hf_upload.click(
                    fn=self._set_data_hf_upload,
                    inputs=[search_in, column_mapping_hf_upload],
                    outputs=self.input_data_component,
                )
            with gradio.Tab(label="Import from file"):
                upload_button = gradio.UploadButton(
                    "Upload",
                    label="Select a file (CSV or Excel)",
                    file_types=["csv", "xlsx", "xlsx"],
                )
                df_upload = gradio.Dataframe(interactive=True)
                upload_button.upload(
                    fn=self.upload_file, inputs=upload_button, outputs=df_upload
                )
                column_mapping_file_upload = gradio.Code(
                    value=json.dumps(dict.fromkeys(self.input_columns, ""), indent=2),
                    **CODE_KWARGS,
                )
                start_btn_file_upload = gradio.Button("Start Annotating")
                start_btn_file_upload.click(
                    fn=self._set_data,
                    inputs=[df_upload, column_mapping_file_upload],
                    outputs=self.input_data_component,
                )

    def _configure_export(self):
        with gradio.Tab("Export data"):
            with gradio.Tab("Export to Hugging Face Hub"):
                with gradio.Row():
                    dataset_name = gradio.Textbox(
                        placeholder="Dataset Name", label="Dataset Name"
                    )
                with gradio.Row():
                    export_button_hf = gradio.Button("Export")
                    export_button_hf.click(
                        fn=self._export_data_hf,
                        inputs=dataset_name,
                        outputs=dataset_name,
                    )
            with gradio.Tab("Export to file"):
                with gradio.Column():
                    export_button = gradio.Button("Export")
                self.file = gradio.File(interactive=False, visible=False)
                export_button.click(
                    fn=self._export_data,
                    outputs=self.file,
                )

    def _set_data_hf_upload(self, repo_id, column_mapping, split="train"):
        gradio.Info("Started loading the dataset. This might take a while.")
        try:
            column_mapping = self._json_to_dict(column_mapping)
            dataset = load_dataset(repo_id, split=split)
            for key, value in column_mapping.items():
                if key != value:
                    if value in dataset.column_names:
                        dataset = dataset.rename_column(value, key)
            dataset = dataset.select_columns(
                [
                    col
                    for col in list(column_mapping.keys())
                    if col in dataset.column_names
                ]
            )
            # add images before converting to bytes
            for column in column_mapping.keys():
                if column in dataset.column_names:
                    self.input_data[column].extend(
                        [self.process_image_input(entry) for entry in dataset[column]]
                    )
            self._set_equal_length_input_data()
            dataframe = pd.DataFrame.from_dict(self.input_data)
            self.start = len(dataframe)
        except Exception as e:
            raise gradio.Error(f"An error occurred: {e}")
        gradio.Info(
            "Data loaded successfully. Showing first 100 examples in 'remaing data' tab. Click on \"⏭️ Next\" to get the next record."
        )
        return dataframe.head(100)

    def _set_data(self, dataframe, column_mapping):
        gradio.Info("Started loading the dataset. This might take a while.")
        try:
            column_mapping = self._json_to_dict(column_mapping)
            dataframe = dataframe[list(column_mapping.values())]
            dataframe.columns = list(column_mapping.keys())
            for column in column_mapping.keys():
                if column in dataframe.columns:
                    self.input_data[column].extend(dataframe[column].tolist())
            self._set_equal_length_input_data()
            dataframe = pd.DataFrame.from_dict(self.input_data)
            self.start = len(dataframe)
        except Exception as e:
            raise gradio.Error(f"An error occurred: {e}")
        gradio.Info(
            "Data loaded successfully. Showing first 100 examples in 'remaing data' tab. Click on 🗑️ discard to get the next record."
        )
        return dataframe.head(100)

    def _export_data_hf(self, dataset_name, oauth_token: gradio.OAuthToken | None):
        gradio.Info("Started exporting the dataset. This may take a while.")
        Dataset.from_dict(self.output_data).push_to_hub(
            dataset_name, token=oauth_token.token
        )
        gradio.Info(f"Exported the dataset to Hugging Face Hub as {dataset_name}.")

    def delete_file_after_delay(self, file_path, delay=30):
        def delete_file():
            time.sleep(delay)
            Path(Path(file_path).name).unlink()

        thread = threading.Thread(target=delete_file)
        thread.start()

    def _export_data(self, dataframe):
        id = uuid.uuid4()
        if "image" in self.task:
            filename = f"{id}.parquet"
            Dataset.from_dict(self.output_data).to_parquet(filename)
        else:
            filename = f"{id}.csv"
            Dataset.from_dict(self.output_data).to_csv(filename)
        self.delete_file_after_delay(filename, 30)
        gradio.Info(
            f"Exported the dataset to {filename}. It will be deleted in 30 seconds."
        )
        return gradio.File(value=filename, visible=True)

    def _delete_file(self, _file):
        return gradio.File(interactive=False, visible=False)

    @staticmethod
    def _json_to_dict(json_str):
        return json.loads(json_str)

    @staticmethod
    def upload_file(file):
        # Determine the file type and load accordingly
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif file.name.endswith(".xls") or file.name.endswith(".xlsx"):
            df = pd.read_excel(file.name)
        else:
            return "Unsupported file type. Please upload a CSV, Excel, or JSON file."
        return df

    def process_image_input(self, input_data):
        if input_data is None:
            return None
        elif isinstance(input_data, dict):
            if "bytes" in input_data and input_data["bytes"]:
                # Case: bytes in a dictionary
                return Image.open(io.BytesIO(input_data["bytes"]))
            elif "path" in input_data and input_data["path"]:
                # Case: path in a dictionary
                return input_data["path"]
        elif isinstance(input_data, Image.Image):
            # Case: PIL Image
            return input_data
        elif isinstance(input_data, str):
            # Case: URL or file path as string
            return input_data
        elif isinstance(input_data, (np.ndarray, list)):
            # Case: numpy array or list
            return Image.fromarray(np.array(input_data))
        else:
            return input_data

    def _set_equal_length_input_data(self):
        # assert all columns for self.input_data are a similar length and fille with "" if not
        max_column_len = max(
            [len(self.input_data[column]) for column in self.input_data.keys()]
        )
        for column in self.input_data.keys():
            if len(self.input_data[column]) < max_column_len:
                self.input_data[column].extend(
                    [""] * (max_column_len - len(self.input_data[column]))
                )
