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

import warnings
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import gradio
import huggingface_hub
from gradio.components import Component

from data_viber._gradio._flagging import FixedHubDatasetSaver
from data_viber._utils import _get_init_payload

if TYPE_CHECKING:
    from transformers.pipelines import Pipeline


class CollectorInterface(gradio.Interface):
    def __init__(
        self,
        fn: Callable,
        inputs: str | Component | Sequence[str | Component] | None,
        outputs: str | Component | Sequence[str | Component] | None,
        *,
        dataset_name: str = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
        allow_flagging: Optional[str] = "auto",
        flagging_options: Optional[List[str]] = None,
        show_embedded_viewer: Optional[bool] = True,
        **kwargs,
    ):
        """
        Load a CollectorInterface with data logging capabilities.

        Args:
            fn: the function to run
            inputs: the input component(s)
            outputs: the output component(s)
            dataset_name: the "org/dataset" to which the data needs to be logged
            hf_token: optional token to pass, otherwise will default to env var HF_TOKEN
            private: whether or not to create a private repo
            allow_flagging: One of "never", "auto", or "manual". If "never" or "auto", users will not see a button to flag an input and output. If "manual", users will see a button to flag. If "auto", every input the user submits will be automatically flagged, along with the generated output. If "manual", both the input and outputs are flagged when the user clicks flag button. This parameter can be set with environmental variable GRADIO_ALLOW_FLAGGING; otherwise defaults to "manual".
            flagging_options: If provided, allows user to select from the list of options when flagging. Only applies if allow_flagging is "manual". Can either be a list of tuples of the form (label, value), where label is the string that will be displayed on the button and value is the string that will be stored in the flagging CSV; or it can be a list of strings ["X", "Y"], in which case the values will be the list of strings and the labels will ["Flag as X", "Flag as Y"], etc.

        Return:
            an intialized CollectorInterface
        """
        self._validate_flagging_options(
            allow_flagging=allow_flagging, flagging_options=flagging_options
        )
        flagging_callback = None or kwargs.pop("flagging_callback", None)
        if dataset_name is not None and flagging_callback is None:
            flagging_callback = kwargs.pop(
                "flagging_callback",
                self._get_flagging_callback(
                    dataset_name=dataset_name, hf_token=hf_token, private=private
                ),
            )
        kwargs.update(
            {
                "flagging_callback": flagging_callback,
                "allow_flagging": allow_flagging,
                "flagging_options": flagging_options,
            }
        )
        super().__init__(fn=fn, inputs=inputs, outputs=outputs, **kwargs)
        self = self._add_html_component_with_viewer(
            self, flagging_callback, show_embedded_viewer
        )

    @classmethod
    def from_pipeline(
        cls,
        pipeline: "Pipeline",
        *,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
        allow_flagging: Optional[str] = "auto",
        flagging_options: Optional[List[str]] = None,
        show_embedded_viewer: Optional[bool] = True,
        **kwargs,
    ) -> gradio.Interface:
        """
        Load an existing transformers.pipeline into a CollectorInterface with data logging capabilities.

        Parameters:
            pipeline: an initialized the transformers.pipeline
            dataset_name: the "org/dataset" to which the data needs to be logged
            hf_token: optional token to pass, otherwise will default to env var HF_TOKEN
            private: whether or not to create a private repo
            allow_flagging: One of "never", "auto", or "manual". If "never" or "auto", users will not see a button to flag an input and output. If "manual", users will see a button to flag. If "auto", every input the user submits will be automatically flagged, along with the generated output. If "manual", both the input and outputs are flagged when the user clicks flag button. This parameter can be set with environmental variable GRADIO_ALLOW_FLAGGING; otherwise defaults to "manual".
            flagging_options: If provided, allows user to select from the list of options when flagging. Only applies if allow_flagging is "manual". Can either be a list of tuples of the form (label, value), where label is the string that will be displayed on the button and value is the string that will be stored in the flagging CSV; or it can be a list of strings ["X", "Y"], in which case the values will be the list of strings and the labels will ["Flag as X", "Flag as Y"], etc.

        Return:
            an intialized CollectorInterface
        """
        return cls.from_interface(
            interface=gradio.Interface.from_pipeline(pipeline=pipeline),
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
            allow_flagging=allow_flagging,
            flagging_options=flagging_options,
            show_embedded_viewer=show_embedded_viewer,
            **kwargs,
        )

    @classmethod
    def from_interface(
        cls,
        interface: gradio.Interface,
        *,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
        allow_flagging: Optional[str] = "auto",
        flagging_options: Optional[List[str]] = None,
        show_embedded_viewer: Optional[bool] = True,
        **kwargs,
    ) -> gradio.Interface:
        """
        Load an existing gradio.Interface into a CollectorInterface with data logging capabilities.

        Parameters:
            interface: any initialized gradio.Interface
            dataset_name: the "org/dataset" to which the data needs to be logged
            hf_token: optional token to pass, otherwise will default to env var HF_TOKEN
            private: whether or not to create a private repo
            allow_flagging: One of "never", "auto", or "manual". If "never" or "auto", users will not see a button to flag an input and output. If "manual", users will see a button to flag. If "auto", every input the user submits will be automatically flagged, along with the generated output. If "manual", both the input and outputs are flagged when the user clicks flag button. This parameter can be set with environmental variable GRADIO_ALLOW_FLAGGING; otherwise defaults to "manual".
            flagging_options: If provided, allows user to select from the list of options when flagging. Only applies if allow_flagging is "manual". Can either be a list of tuples of the form (label, value), where label is the string that will be displayed on the button and value is the string that will be stored in the flagging CSV; or it can be a list of strings ["X", "Y"], in which case the values will be the list of strings and the labels will ["Flag as X", "Flag as Y"], etc.

        Return:
            an intialized CollectorInterface
        """
        flagging_callback = None or kwargs.pop("flagging_callback", None)
        if dataset_name and not flagging_callback:
            flagging_callback = cls._get_flagging_callback(
                dataset_name=dataset_name, hf_token=hf_token, private=private
            )
        payload = _get_init_payload(interface)
        payload.update(**kwargs)
        payload.update(
            {
                "flagging_callback": flagging_callback,
                "allow_flagging": allow_flagging,
                "flagging_options": flagging_options,
                "show_embedded_viewer": show_embedded_viewer,
            }
        )
        return cls(**payload)

    @staticmethod
    def _validate_flagging_options(allow_flagging, flagging_options) -> None:
        if allow_flagging == "auto" and flagging_options:
            raise ValueError(
                "automatic flagging cannot be combined with 'flagging_options', set `allow_flagging='manual'` instead"
            )
        if allow_flagging == "never":
            warnings.warn("You are using a datacollector but don't enable flagging")

    @staticmethod
    def _get_flagging_callback(
        dataset_name: str,
        hf_token: str,
        private: bool = False,
    ) -> gradio.HuggingFaceDatasetSaver:
        return FixedHubDatasetSaver(
            hf_token=hf_token,
            dataset_name=dataset_name,
            private=private,
            info_filename="dataset_info.json",
            separate_dirs=True,
        )

    @staticmethod
    def _get_repo_url(
        flagging_callback: gradio.HuggingFaceDatasetSaver,
    ) -> huggingface_hub.RepoUrl:
        repo_id = huggingface_hub.create_repo(
            repo_id=flagging_callback.dataset_id,
            token=flagging_callback.hf_token,
            private=flagging_callback.dataset_private,
            repo_type="dataset",
            exist_ok=True,
        ).repo_id
        return f"https://huggingface.co/datasets/{repo_id}"

    @staticmethod
    def _get_embedded_dataset_viewer(repo_url: str) -> str:
        return f"""
                <iframe
                src="{repo_url}/embed/viewer/default/train"
                frameborder="0"
                width="100%"
                height="560px"
                ></iframe>
                """

    @classmethod
    def _add_html_component_with_viewer(
        cls,
        instance: gradio.Interface,
        flagging_callback: Optional[gradio.HuggingFaceDatasetSaver] = None,
        show_embedded_viewer: bool = True,
    ):
        if flagging_callback:
            repo_url = cls._get_repo_url(flagging_callback)
            formatted_repo_url = (
                f"Data is being written to [a dataset on the Hub]({repo_url})."
            )
            with instance:
                with gradio.Row(equal_height=False):
                    gradio.Markdown(formatted_repo_url)
                if show_embedded_viewer and not flagging_callback.dataset_private:
                    with gradio.Row():
                        with gradio.Accordion(
                            "dataset viewer - do an (empty) search to refresh",
                            open=False,
                        ):
                            gradio.HTML(cls._get_embedded_dataset_viewer(repo_url))
        else:
            with instance:
                gradio.Info("Data is stored locally in a CSV file")
        return instance
