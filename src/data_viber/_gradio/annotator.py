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

from typing import List, Optional, override

import gradio
from gradio.components import (
    Button,
    ClearButton,
)
from gradio.events import Dependency
from gradio.flagging import FlagMethod

from data_viber._gradio.collector import GradioDataCollectorInterface

_DEFAULT_COLORS = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]


class GradioAnnotatorInterFace(GradioDataCollectorInterface):
    @classmethod
    def for_text_classification(
        cls,
        texts: List[str],
        labels: List[str],
        *,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
        multi_label: Optional[bool] = False,
    ) -> "GradioAnnotatorInterFace":
        start = len(texts)

        def next_input(annotate, checkboxgroup):
            if texts:
                gradio.Info(
                    f"{(len(texts) / start) * 100:.2f}% done {len(texts)} left."
                )
                text = texts.pop()
                return (text, []) if multi_label else text
            else:
                gradio.Info("No data to annotate left.")
                return ("", []) if multi_label else ""

        inputs = gradio.TextArea(value=texts.pop(), label="text")
        if multi_label:
            inputs = [inputs, gradio.CheckboxGroup(labels, label="labels")]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=[("✍🏼 submit", "")]
            if multi_label
            else [(lab, lab) for lab in labels],
            allow_flagging="manual",
            submit_btn=gradio.Button("✍🏼 submit", variant="primary", visible=False),
            clear_btn=gradio.Button("🗑️ discard", variant="stop"),
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_token_classification(
        cls,
        texts: List[str],
        labels: List[str],
        *,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "GradioAnnotatorInterFace":
        if isinstance(labels, list):
            labels = {label: color for label, color in zip(labels, _DEFAULT_COLORS)}

        def convert_to_tokens(text: str):
            return [(char, None) for char in text]

        start = len(texts)

        def next_input(_):
            if texts:
                gradio.Info(
                    f"{(len(texts) / start) * 100:.2f}% done {len(texts)} left."
                )
                return convert_to_tokens(texts.pop())
            else:
                gradio.Info("No data to annotate left.")
                return ""

        input_text = gradio.HighlightedText(
            value=convert_to_tokens(texts.pop()),
            color_map=labels,
            label="spans",
            interactive=True,
            show_legend=False,
            combine_adjacent=True,
            adjacent_separator="",
        )
        return cls(
            fn=next_input,
            inputs=[input_text],
            outputs=[input_text],
            submit_btn=gradio.Button("✍🏼 submit", variant="primary", visible=False),
            clear_btn=gradio.Button("🗑️ discard", variant="stop"),
            flagging_options=[("✍🏼 submit", "")],
            allow_flagging="manual",
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_question_answering(
        cls,
        questions: List[str],
        contexts: List[str],
        *,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "GradioAnnotatorInterFace":
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must be of the same length")
        start = len(questions)

        def next_input(a, b):
            try:
                gradio.Info(
                    f"{(len(questions) / start) * 100:.2f}% done {len(questions)} left."
                )
                return questions.pop(), contexts.pop()
            except IndexError:
                gradio.Info("No data to annotate left")
                return None, None

        input_question = gradio.TextArea(value=questions.pop(), label="question")
        input_context = gradio.HighlightedText(
            value=contexts.pop(),
            label="context",
            interactive=True,
            show_legend=False,
            adjacent_separator="",
        )
        return cls(
            fn=next_input,
            inputs=[input_question, input_context],
            outputs=[input_question, input_context],
            allow_flagging="manual",
            submit_btn=gradio.Button("✍🏼 submit", variant="primary", visible=False),
            clear_btn=gradio.Button("🗑️ discard", variant="stop"),
            flagging_options=[("✍🏼 submit", "")],
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_chat_preference(cls):
        raise NotImplementedError

    @override
    def attach_flagging_events(
        self,
        flag_btns: list[Button, gradio.CheckboxGroup] | None,
        _clear_btn: ClearButton,
        _submit_event: Dependency,
    ):
        # before the flaffing because otherwise input is reset
        self.attach_submit_events(_submit_btn=_clear_btn, _stop_btn=None)
        super().attach_flagging_events(flag_btns, _clear_btn, _submit_event)
        if self.allow_flagging == "manual":
            for flag_btn in flag_btns:
                if flag_btn.label != "✍🏼 submit":
                    self.attach_submit_events(_submit_btn=flag_btn, _stop_btn=None)
            if flag_btn.label == "✍🏼 submit":
                flag_method = FlagMethod(
                    self.flagging_callback, "", "", visual_feedback=False
                )
                flag_btn.click(
                    flag_method,
                    inputs=self.input_components + self.output_components,
                    outputs=None,
                    preprocess=False,
                    queue=False,
                    show_api=False,
                )

    @override
    def render_flag_btns(self) -> list[Button]:
        return [Button(label, variant="primary") for label, _ in self.flagging_options]
