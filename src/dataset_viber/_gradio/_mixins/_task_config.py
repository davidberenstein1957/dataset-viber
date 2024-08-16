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

import gradio


class TaskConfigMixin:
    def _set_text_classification_config(self, inputs):
        if self.task in [
            "text-classification",
            "text-classification-multi-label",
            "chat-classification",
            "chat-classification-multi-label",
            "image-classification",
            "image-classification-multi-label",
        ]:
            with gradio.Tab("Label selector"):
                with gradio.Column():
                    label_selector = gradio.Dropdown(
                        choices=[],
                        label="label",
                        allow_custom_value=True,
                        multiselect=True,
                    )

                def update_labels(_label_selector):
                    self.labels = _label_selector
                    _kwargs = {
                        "choices": _label_selector,
                        "label": "label",
                    }
                    return (
                        gradio.CheckboxGroup(**_kwargs)
                        if "multi-label" in self.task
                        else gradio.CheckboxGroup(**_kwargs)
                    )

                label_selector.change(
                    fn=update_labels,
                    inputs=[label_selector],
                    outputs=[
                        input
                        for input in inputs
                        if isinstance(input, (gradio.Radio, gradio.CheckboxGroup))
                    ],
                )
