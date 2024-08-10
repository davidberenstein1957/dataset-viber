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

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, override

import gradio as gr
import huggingface_hub
from gradio import utils
from gradio.flagging import HuggingFaceDatasetSaver
from gradio_client import utils as client_utils


class FixedHubDatasetSaver(HuggingFaceDatasetSaver):
    @override
    def _deserialize_components(
        self,
        data_dir: Path,
        flag_data: list[Any],
        flag_option: str = "",
        username: str = "",
    ) -> tuple[dict[Any, Any], list[Any]]:
        """Deserialize components and return the corresponding row for the flagged sample.

        Images/audio are saved to disk as individual files.
        """
        # Components that can have a preview on dataset repos
        file_preview_types = {gr.Audio: "Audio", gr.Image: "Image"}

        # Generate the row corresponding to the flagged sample
        features = OrderedDict()
        row = []
        for component, sample in zip(self.components, flag_data):
            # Get deserialized object (will save sample to disk if applicable -file, audio, image,...-)
            label = component.label or ""
            save_dir = data_dir / client_utils.strip_invalid_filename_characters(label)
            save_dir.mkdir(exist_ok=True, parents=True)
            deserialized = utils.simplify_file_data_in_str(
                component.flag(sample, save_dir)
            )

            # Add deserialized object to row
            features[label] = {"dtype": "string", "_type": "Value"}
            try:
                deserialized_path = Path(deserialized)
                if not deserialized_path.exists():
                    raise FileNotFoundError(f"File {deserialized} not found")
                row.append(str(deserialized_path.relative_to(self.dataset_dir)))
            except (FileNotFoundError, TypeError, ValueError, OSError):
                deserialized = "" if deserialized is None else str(deserialized)
                row.append(deserialized)

            # If component is eligible for a preview, add the URL of the file
            # Be mindful that images and audio can be None
            if isinstance(component, tuple(file_preview_types)):  # type: ignore
                for _component, _type in file_preview_types.items():
                    if isinstance(component, _component):
                        features[label + " file"] = {"_type": _type}
                        break
                if deserialized:
                    path_in_repo = str(  # returned filepath is absolute, we want it relative to compute URL
                        Path(deserialized).relative_to(self.dataset_dir)
                    ).replace("\\", "/")
                    row.append(
                        huggingface_hub.hf_hub_url(
                            repo_id=self.dataset_id,
                            filename=path_in_repo,
                            repo_type="dataset",
                        )
                    )
                else:
                    row.append("")
        features["flag"] = {"dtype": "string", "_type": "Value"}
        features["username"] = {"dtype": "string", "_type": "Value"}
        row.append(flag_option)
        row.append(username)
        return features, row
