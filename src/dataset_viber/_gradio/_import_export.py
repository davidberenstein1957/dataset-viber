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

import json
import uuid
from pathlib import Path

import gradio
import pandas as pd
from datasets import Dataset, load_dataset
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from huggingface_hub import whoami

CODE_KWARGS = {
    "language": "json",
    "interactive": True,
    "label": "Column Mapping",
    "lines": 1,
}


class ImportExportMixin:
    def _list_organizations(self, oauth_token: gradio.OAuthToken | None) -> str:
        orgs = []
        if oauth_token is not None:
            orgs = [str(org["name"]) for org in whoami(oauth_token.token)["orgs"]]
        return ",".join(orgs)

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
        with self:
            with gradio.Accordion("Import data", open=False):
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
                        value=json.dumps(
                            dict.fromkeys(self.input_columns, ""), indent=2
                        ),
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
                        value=json.dumps(
                            dict.fromkeys(self.input_columns, ""), indent=2
                        ),
                        **CODE_KWARGS,
                    )
                    start_btn_file_upload = gradio.Button("Start Annotating")
                    start_btn_file_upload.click(
                        fn=self._set_data,
                        inputs=[df_upload, column_mapping_file_upload],
                        outputs=self.input_data_component,
                    )

    def _configure_export(self):
        with self:
            with gradio.Accordion("Export to Hugging Face", open=False):
                with gradio.Tab("Export to Hugging Face Hub"):
                    with gradio.Row():
                        with gradio.Column():
                            organization = gradio.Textbox(label="Organization")
                            self.load(self._list_organizations, outputs=organization)
                        with gradio.Column():
                            dataset_name = gradio.Textbox(
                                placeholder="Dataset Name", label="Dataset Name"
                            )
                    with gradio.Row():
                        export_button_hf = gradio.Button("Export")
                        export_button_hf.click(
                            fn=self._export_data_hf,
                            inputs=[self.output_data_component, dataset_name],
                            outputs=dataset_name,
                        )
                with gradio.Tab("Export to file"):
                    with gradio.Column():
                        export_button = gradio.Button("Export")
                    with gradio.Column():
                        delete_button = gradio.Button("🗑️ delete file")
                    self.file = gradio.File(interactive=False, visible=False)
                    export_button.click(
                        fn=self._export_data,
                        inputs=self.output_data_component,
                        outputs=self.file,
                    )
                    delete_button.click(
                        self._delete_file, inputs=self.file, outputs=self.file
                    )

    def _set_data_hf_upload(self, repo_id, column_mapping, split="train"):
        gradio.Info("Started loading the dataset. This may take a while.")
        dataset = load_dataset(repo_id, split=split)
        dataframe = dataset.to_pandas()
        return self._set_data(dataframe, column_mapping)

    def _set_data(self, dataframe, column_mapping):
        column_mapping = self._load_json_as_dict(column_mapping)
        dataframe = dataframe[list(column_mapping.values())]
        dataframe.columns = list(column_mapping.keys())
        for column in column_mapping.keys():
            self.input_data[column].extend(dataframe[column].tolist())
        gradio.Info(
            "Data loaded successfully. Click on 🗑️ discard to get the next record."
        )
        return dataframe

    def _export_data_hf(self, dataframe: pd.DataFrame, dataset_name):
        gradio.Info("Started exporting the dataset. This may take a while.")
        Dataset.from_pandas(dataframe).push_to_hub(dataset_name)
        gradio.Info(f"Exported the dataset to Hugging Face Hub as {dataset_name}.")
        raise ""

    def _create_dataset_card(self, repo_id):
        pass

    def _get_autotrain_config(self):
        pass

    def _get_argilla_config(self):
        pass

    def _export_data(self, dataframe):
        id = uuid.uuid4()
        filename = f"{id}.csv"
        dataframe.to_csv(filename, index=False)
        return gradio.File(value=filename, visible=True)

    def _delete_file(self, _file):
        self.file.delete()
        Path(Path(_file).name).unlink()
        return gradio.File(interactive=False, visible=False)

    @staticmethod
    def _load_json_as_dict(json_str):
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
