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

import random
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import gradio
import numpy as np
import pandas as pd
import PIL
from gradio.components import (
    Button,
    ClearButton,
)
from gradio.events import Dependency
from gradio.flagging import FlagMethod

from dataset_viber._gradio._mixins._import_export import ImportExportMixin
from dataset_viber._gradio._mixins._task_config import TaskConfigMixin
from dataset_viber._gradio.collector import CollectorInterface

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

_POP_INDEX = 0
_MESSAGE_DONE_ANNOTATING = "No data left to annotate."
_HIGHLIGHT_TEXT_KWARGS = {
    "interactive": True,
    "show_legend": True,
    "combine_adjacent": True,
    "adjacent_separator": "",
}
_CHATBOT_KWARGS = {"type": "messages", "label": "prompt", "show_copy_button": True}
_SUBMIT_BTN = gradio.Button("✍🏼 submit", variant="primary", visible=False)
_CLEAR_BTN = gradio.Button("🗑️ discard", variant="stop")
_PREFERENCE_OPTIONS = [("👆 A is better", "A"), ("👇 B is better", "B")]
_SUBMIT_OPTIONS = [("✍🏼 submit", "")]

if TYPE_CHECKING:
    from transformers.pipelines import Pipeline


class AnnotatorInterFace(CollectorInterface, ImportExportMixin, TaskConfigMixin):
    @override
    def __init__(
        self,
        *args,
        inputs,
        outputs,
        **kwargs,
    ):
        self._override_block_init_method(**kwargs)
        with self:
            gradio.LoginButton(
                value="Sign in with Hugging Face - a login will reset the data!"
            ).activate()
            with gradio.Accordion(
                open=False if self.start else True, label="Import and remaining data"
            ):
                with gradio.Tab("Remaining data"):
                    self.input_data_component = gradio.Dataframe(
                        pd.DataFrame.from_dict(self.input_data).head(100),
                        interactive=False,
                    )
                    inputs[0].change(
                        fn=lambda x: pd.DataFrame.from_dict(self.input_data).head(100),
                        outputs=self.input_data_component,
                    )
                self._set_text_classification_config(inputs)
                self._configure_import()
        super().__init__(
            *args,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        with self:
            with gradio.Row():
                # with gradio.Column():
                #     sort = gradio.Button("Semantic Sort", variant="secondary")
                with gradio.Column():
                    shuffle = gradio.Button("Shuffle", variant="secondary")
                # sort.click(fn=self.sort_data, inputs=None, outputs=None)
                shuffle.click(
                    fn=self.shuffle_data, inputs=None, outputs=self.input_data_component
                )
            with gradio.Accordion(open=False, label="Export and completed data"):
                with gradio.Tab("Completed data"):
                    self.output_data_component = gradio.Dataframe(
                        pd.DataFrame.from_dict(self.output_data).tail(100),
                        interactive=False,
                    )
                    inputs[0].change(
                        fn=lambda x: pd.DataFrame.from_dict(self.output_data).tail(100),
                        outputs=self.output_data_component,
                    )
                self._configure_export()

    @classmethod
    def for_text_classification(
        cls,
        texts: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        multi_label: Optional[bool] = False,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for text classification tasks.

        Parameters:
            texts (Optional[List[str]]): List of texts to annotate.
            suggestions (Optional[List[str]]): List of suggestions to correct for. Defaults to None.
            labels (Optional[List[str]]): List of labels to choose from.
            multi_label (Optional[bool]): Whether to allow multiple labels. Defaults to False.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the text before annotating.
                Expecting it takes a `str` and returns [{"label": str, "score": float}].
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = (
            "text-classification"
            if not multi_label
            else "text-classification-multi-label"
        )
        cls.labels = labels or []
        cls.input_columns = ["text", "suggestion"]
        cls.output_columns = ["text", "label"]
        cls.input_data = {"text": texts or [], "suggestion": suggestions or []}
        cls.output_data = {label: [] for label in cls.output_columns}

        # Process function
        cls.start = len(cls.input_data["text"])

        def next_input(_text, _label):
            if _text:
                cls.output_data["text"].append(_text)
                cls.output_data["label"].append(_label)
            if cls.input_data["text"]:
                cls._update_message(cls)
                text = cls.input_data["text"].pop(_POP_INDEX)
                label = ""
                if "suggestion" in cls.input_data:
                    label = cls.input_data["suggestion"].pop(_POP_INDEX)
                label = label if fn is None or label else fn(text)
                if cls.task == "text-classification-multi-label":
                    label = [lab["label"] for lab in label if lab["score"] > 0.5]
                    return (text, label)
                elif cls.task == "text-classification" and fn is not None:
                    return (text, label[0]["label"])
                else:
                    return (text, [] if multi_label else None)
            else:
                cls._done_message()
                return ("", [])

        # UI Config
        text, label = next_input(None, None)
        if cls.task == "text-classification-multi-label":
            check_box_group = gradio.CheckboxGroup(
                cls.labels, value=label, label="label"
            )
        else:
            check_box_group = gradio.Radio(cls.labels, value=label, label="label")
        inputs: List[gradio.Textbox] = [
            gradio.Textbox(value=text, label="text"),
            check_box_group,
        ]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=_SUBMIT_OPTIONS,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_token_classification(
        cls,
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for token classification tasks.

        Parameters:
            texts (Optional[List[str]]): List of texts to annotate.
            labels (Optional[List[str]]): List of labels to choose from.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the text before annotating.
                Expecting it takes a `str` and returns List[Tuple[str, str]] [("text","label")].
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "token-classification"
        cls.labels = labels or []
        cls.input_columns = ["text"]
        cls.output_columns = ["text", "spans"]
        cls.input_data = {"text": texts or []}
        cls.output_data = {label: [] for label in cls.output_columns}

        # Process function
        cls.start = len(cls.input_data["text"])

        def next_input(_text, _spans):
            if _text:
                cls.output_data["text"].append(_text)
                cls.output_data["spans"].append(_spans)
            if cls.input_data["text"]:
                cls._update_message(cls)
                text = cls.input_data["text"].pop(_POP_INDEX)
                spans = cls._convert_to_tokens(text) if fn is None else fn(text)
                return text, spans
            else:
                cls._done_message()
                return "", cls._convert_to_tokens(" ")

        # UI Config
        text, spans = next_input(None, None)
        inputs = [
            gradio.Textbox(value=text, label="text", interactive=False),
            gradio.HighlightedText(
                value=spans,
                label="spans",
                **_HIGHLIGHT_TEXT_KWARGS,
            ),
        ]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            flagging_options=_SUBMIT_OPTIONS,
            allow_flagging="manual",
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_question_answering(
        cls,
        questions: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for question answering tasks.

        Parameters:
            questions (Optional[List[str]]): List of questions to annotate.
            contexts (Optional[List[str]]): List of contexts to annotate.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the context before annotating.
                Expecting it takes a `str` and returns [{"label": str, "score": float}].
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "question-answering"
        cls.input_columns = ["question", "context"]
        cls.output_columns = ["question", "context"]
        cls.input_data = {"question": questions or [], "context": contexts or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["question"])

        # Input validation
        cls.start = len(cls.input_data["question"])
        if cls.input_data["question"] and cls.input_data["context"]:
            if len(cls.input_data["question"]) != len(cls.input_data["context"]):
                raise ValueError("Questions and contexts must be of the same length.")

        # Process function
        def next_input(_question, _context):
            if _question:
                cls.output_data["question"].append(_question)
                cls.output_data["context"].append(_context)
            if cls.input_data["question"]:
                cls._update_message(cls)
                question = cls.input_data["question"].pop(_POP_INDEX)
                context = cls.input_data["context"].pop(_POP_INDEX)
                context = (
                    cls._convert_to_tokens(context)
                    if fn is None
                    else fn(question, context)
                )
                return question, context
            else:
                cls._done_message()
                return None, []

        # UI Config
        question, context = next_input(None, None)
        input_question = gradio.Textbox(value=question, label="question")
        input_context = gradio.HighlightedText(
            value=context,
            label="context",
            interactive=True,
            show_legend=False,
            combine_adjacent=True,
            adjacent_separator="",
        )
        inputs = [input_question, input_context]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            flagging_options=_SUBMIT_OPTIONS,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_text_generation(
        cls,
        prompts: Optional[List[str]] = None,
        completions: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for text generation tasks.

        Parameters:
            prompts (Optional[List[str]]): List of prompts to annotate.
            completions (Optional[List[str]]): List of completions to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the prompt before annotating.
                Expecting it takes a `str` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "text-generation"
        cls.input_columns = ["prompt", "completion"]
        cls.output_columns = ["prompt", "completion"]
        cls.input_data = {"prompt": prompts or [], "completion": completions or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["prompt"])

        # Input validation
        if cls.input_data["prompt"] and cls.input_data["completion"]:
            if len(cls.input_data["prompt"]) != len(cls.input_data["completion"]):
                raise ValueError(
                    "Source and target must be of the same length. You can add empty strings to match the lengths."
                )

        # Process function
        def next_input(_prompt, _completion):
            if _prompt:
                cls.output_data["prompt"].append(_prompt)
                cls.output_data["completion"].append(_completion)
            if cls.input_data["prompt"]:
                cls._update_message(cls)
                prompt = cls.input_data["prompt"].pop(_POP_INDEX)
                completion = (
                    cls.output_data["completion"].pop(_POP_INDEX)
                    if fn is None
                    else fn(prompt)
                )
                return prompt, completion
            else:
                cls._done_message()
                return None, None

        # UI Config
        prompt, completion = next_input(None, None)
        input_prompt = gradio.Textbox(value=prompt, label="prompt")
        input_completion = gradio.Textbox(value=completion, label="completion")
        inputs = [input_prompt, input_completion]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            flagging_options=_SUBMIT_OPTIONS,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_text_generation_preference(
        cls,
        prompts: Optional[List[str]] = None,
        completions_a: Optional[List[str]] = None,
        completions_b: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for text generation preference tasks.

        Parameters:
            prompts (Optional[List[str]]): List of prompts to annotate.
            completions_a (Optional[List[str]]): List of completions to annotate for option A. Defaults to None.
            completions_b (Optional[List[str]]): List of completions to annotate for option B. Defaults to None.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the prompt before annotating.
                Expecting it takes a `str` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "text-generation-preference"
        cls.input_columns = ["prompt", "completion_a", "completion_b"]
        cls.output_columns = ["prompt", "chosen", "rejected", "flag"]
        cls.input_data = {
            "prompt": prompts or [],
            "completion_a": completions_a or [],
            "completion_b": completions_b or [],
        }
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["prompt"])

        # Input validation
        (
            cls.input_data["promt"],
            cls.input_data["completion_a"],
            cls.input_data["completion_b"],
        ) = cls._validate_preference(
            fn,
            cls.input_data["promt"],
            cls.input_data["completion_a"],
            cls.input_data["completion_b"],
        )

        # Process function
        def next_input(_prompt, _completion_a, _completion_b):
            if _prompt:
                cls.output_data["prompt"].append(_prompt)
                if cls.output_data["flag"][-1] == "👆 A is better":
                    cls.output_data["chosen"].append(_completion_a)
                    cls.output_data["rejected"].append(_completion_b)
                else:
                    cls.output_data["chosen"].append(_completion_b)
                    cls.output_data["rejected"].append(_completion_a)
            if cls.input_data["prompt"]:
                cls._update_message(cls)
                prompt = cls.input_data["prompt"].pop(_POP_INDEX)
                completion_a = cls.input_data["completion_a"].pop(_POP_INDEX)
                completion_b = cls.input_data["completion_b"].pop(_POP_INDEX)
                completion_a = (
                    completion_a if fn is None or completion_a != "" else fn(prompt)
                )
                completion_b = (
                    completion_b if fn is None or completion_b != "" else fn(prompt)
                )
                return prompt, completion_a, completion_b
            else:
                cls._done_message()
                return None, None, None

        # UI Config
        prompt, completion_a, completion_b = next_input(None, None, None)
        input_prompt = gradio.Textbox(value=prompt, label="prompt")
        input_completion_a = gradio.Textbox(value=completion_a, label="👆 completion A")
        input_completion_b = gradio.Textbox(value=completion_b, label="👇 completion B")
        inputs = [input_prompt, input_completion_a, input_completion_b]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            flagging_options=_PREFERENCE_OPTIONS,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_chat_classification(
        cls,
        prompts: Optional[List[List[Dict[str, str]]]] = None,
        suggestions: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        multi_label: Optional[bool] = False,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for chat classification tasks.

        Parameters:
            prompts (Optional[List[List[Dict[str, str]]]): List of chat messages to annotate.
            suggestions (Optional[List[str]]): List of suggestions to correct for. Defaults to None.
            labels (Optional[List[str]]): List of labels to choose from.
            multi_label (Optional[bool]): Whether to allow multiple labels. Defaults to False.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the chat messages before annotating.
                Expecting it takes a `str` and returns [{"label": str, "score": float}].
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = (
            "chat-classification"
            if not multi_label
            else "chat-classification-multi-label"
        )
        cls.labels = labels
        cls.input_columns = ["prompt", "suggestion"]
        cls.output_columns = ["prompt", "label"]
        cls.input_data = {"prompt": prompts or [], "suggestion": suggestions or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["prompt"])

        # Input validation
        cls.input_data["prompt"] = cls._convert_to_chat_message(
            cls.input_data["prompt"]
        )

        # Process function
        def next_input(_prompt, _label):
            if _prompt:
                cls.output_data["prompt"].append(_prompt)
                cls.output_data["label"].append(_label)
            if cls.input_data["prompt"]:
                cls._update_message(cls)
                prompt = cls.input_data["prompt"].pop(_POP_INDEX)
                label = ""
                if "suggestion" in cls.input_data:
                    label = cls.input_data["suggestion"].pop(_POP_INDEX)
                label = (
                    label
                    if fn is None or label
                    else fn("\n".join([msg.content for msg in prompt]))
                )
                if cls.task == "chat-classification-multi-label":
                    label = [lab["label"] for lab in label if lab["score"] > 0.5]
                    return (prompt, label)
                elif cls.task == "chat-classification" and fn is not None:
                    return (prompt, label[0]["label"])
                else:
                    return (prompt, [] if multi_label else "")
            else:
                cls._done_message()
                return (None, [])

        # UI Config
        prompt, label = next_input(None, None)
        if cls.task == "chat-classification-multi-label":
            check_box_group = gradio.CheckboxGroup(
                cls.labels, value=label, label="label"
            )
        else:
            check_box_group = gradio.Radio(cls.labels, value=label, label="label")
        inputs = [
            gradio.Chatbot(
                value=prompt,
                **_CHATBOT_KWARGS,
            ),
            check_box_group,
        ]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=_SUBMIT_OPTIONS,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_chat_generation(
        cls,
        prompts: Optional[
            Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]]
        ] = None,
        completions: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for chat generation tasks.

        Parameters:
            prompts (Optional[Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]]): List of chat messages to annotate.
            completions (Optional[List[str]]): List of completions to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the chat messages before annotating.
                Expecting it takes a `List[gradio.ChatMessage]` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "chat-generation"
        cls.input_columns = ["prompt", "completion"]
        cls.output_columns = ["prompt", "completion"]
        cls.input_data = {"prompt": prompts or [], "completion": completions or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["prompt"])

        # Input validation
        if cls.input_data["prompt"] and cls.input_data["completion"]:
            if len(cls.input_data["prompt"]) != len(cls.input_data["completion"]):
                raise ValueError(
                    "Source and target must be of the same length. You can add empty strings to match the lengths."
                )
        cls.input_data["prompt"] = cls._convert_to_chat_message(
            cls.input_data["prompt"]
        )

        # Process function
        def next_input(_prompt, _completion):
            def _last_is_user(_prompt):
                return _prompt[-1].role == "user"

            if _prompt:
                cls.output_data["prompt"].append(_prompt)
                cls.output_data["completion"].append(_completion)
            if cls.input_data["prompt"]:
                cls._update_message(cls)
                prompt = cls.input_data["prompt"].pop(_POP_INDEX)
                completion = cls.input_data["completion"].pop(_POP_INDEX)
                completion = (
                    completion
                    if (fn is None or completion != "") and _last_is_user(prompt)
                    else fn(prompt)
                )
                return prompt, completion
            else:
                cls._done_message()
                return [], None

        # UI Config
        prompt, completion = next_input(None, None)
        input_prompt = gradio.Chatbot(
            value=prompt,
            **_CHATBOT_KWARGS,
        )
        input_completion = gradio.Textbox(value=completion, label="completion")
        inputs = [input_prompt, input_completion]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            flagging_options=_SUBMIT_OPTIONS,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_chat_generation_preference(
        cls,
        prompts: Optional[
            Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]]
        ] = None,
        completions_a: Optional[List[str]] = None,
        completions_b: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for chat generation preference tasks.

        Parameters:
            prompts (Optional[Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]]): List of chat messages to annotate.
            completions_a (Optional[List[str]]): List of completions to annotate for option A.
            completions_b (Optional[List[str]]): List of completions to annotate for option B.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the chat messages before annotating.
                Expecting it takes a `List[gradio.ChatMessage]` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "chat-generation-preference"
        cls.input_columns = ["prompt", "completion_a", "completion_b"]
        cls.output_columns = ["prompt", "chosen", "rejected", "flag"]
        cls.input_data = {
            "prompt": prompts or [],
            "completion_a": completions_a or [],
            "completion_b": completions_b or [],
        }
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["prompt"])

        # Input validation
        (
            cls.input_data["promt"],
            cls.input_data["completion_a"],
            cls.input_data["completion_b"],
        ) = cls._validate_preference(
            fn,
            cls.input_data["promt"],
            cls.input_data["completion_a"],
            cls.input_data["completion_b"],
        )
        cls.input_data["prompt"] = cls._convert_to_chat_message(
            cls.input_data["prompt"]
        )

        # Process function
        def next_input(_prompt, _completion_a, _completion_b):
            if _prompt:
                cls.output_data["prompt"].append(_prompt)
                if cls.output_data["flag"][-1] == "👆 A is better":
                    cls.output_data["chosen"].append(_completion_a)
                    cls.output_data["rejected"].append(_completion_b)
                else:
                    cls.output_data["chosen"].append(_completion_b)
                    cls.output_data["rejected"].append(_completion_a)
            if cls.input_data["prompt"]:
                cls._update_message(cls)
                prompt = cls.input_data["prompt"].pop(_POP_INDEX)
                completion_a = cls.input_data["completion_a"].pop(_POP_INDEX)
                completion_b = cls.input_data["completion_b"].pop(_POP_INDEX)
                completion_a = (
                    completion_a if fn is None or completion_a != "" else fn(prompt)
                )
                completion_b = (
                    completion_b if fn is None or completion_b != "" else fn(prompt)
                )
                return prompt, completion_a, completion_b
            else:
                cls._done_message()
                return [], None, None

        # UI Config
        prompt, completion_a, completion_b = next_input(None, None, None)
        input_prompt = gradio.Chatbot(value=prompt, **_CHATBOT_KWARGS)
        input_completion_a = gradio.Textbox(value=completion_a, label="👆 completion A")
        input_completion_b = gradio.Textbox(value=completion_b, label="👇 completion B")
        inputs = [input_prompt, input_completion_a, input_completion_b]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            flagging_options=_PREFERENCE_OPTIONS,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_image_classification(
        cls,
        images: Optional[List[Union[np.array, PIL.Image.Image, str]]] = None,
        suggestions: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        multi_label: Optional[bool] = False,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image classification tasks.

        Parameters:
            images (List[Union[np.array, PIL.Image.Image, str]]): List of images to annotate.
            labels (List[str]): List of labels to choose from.
            multi_label (Optional[bool]): Whether to allow multiple labels. Defaults to False.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the image before annotating.
                it should take an `np.array` or `PIL.Image.Image` and return `Union[str, List[Dict[str, Union[str, float]]]]`.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = (
            "image-classification"
            if not multi_label
            else "image-classification-multi-label"
        )
        cls.labels = labels or []
        cls.input_columns = ["image", "suggestion"]
        cls.output_columns = ["image", "label"]
        cls.input_data = {"image": images or [], "suggestion": suggestions or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["image"])

        # Process function
        def next_input(_image, _label):
            if _image:
                cls.output_data["image"].append(_image)
                cls.output_data["label"].append(_label)
            if cls.input_data["image"]:
                cls._update_message(cls)
                image = cls.input_data["image"].pop(_POP_INDEX)
                label = cls.input_data["suggestion"].pop(_POP_INDEX)
                label = label if fn is None or label else fn(image)
                if cls.task == "image-classification-multi-label":
                    label = [lab["label"] for lab in label if lab["score"] > 0.5]
                    return (image, label)
                elif cls.task == "image-classification" and fn is not None:
                    return (image, label[0]["label"])
                else:
                    return (image, [] if multi_label else None)
            else:
                cls._done_message()
                return (None, [])

        # UI Config
        image, label = next_input(None, None)
        inputs = gradio.Image(value=image, label="image", height=400)
        if cls.task == "image-classification-multi-label":
            check_box_group = gradio.CheckboxGroup(
                cls.labels, value=label, label="label"
            )
        else:
            check_box_group = gradio.Radio(cls.labels, value=label, label="label")
        inputs = [inputs, check_box_group]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=_SUBMIT_OPTIONS,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_image_generation(
        cls,
        prompts: Optional[List[str]] = None,
        completions: Optional[List[Union[np.array, PIL.Image.Image, str]]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image generation tasks.

        Parameters:
            prompts (Optional[List[str]]): List of prompts to annotate.
            completions (Optional[List[Union[np.array, PIL.Image.Image, str]]]): List of completions to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the prompt to generate an image.
                it takes a `str` and returns `Union[np.array, PIL.Image.Image, str]`.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "image-generation"
        cls.input_columns = ["prompt", "completion"]
        cls.output_columns = ["prompt", "completion"]
        cls.input_data = {"prompt": prompts or [], "completion": completions or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["prompt"])

        # Input validation
        if cls.input_data["prompt"] and cls.input_data["completion"]:
            if len(cls.input_data["prompt"]) != len(cls.input_data["completion"]):
                raise ValueError(
                    "Source and target must be of the same length. You can add empty strings to match the lengths."
                )

        # Process function
        def next_input(_prompt, _completion):
            if _prompt:
                cls.output_data["prompt"].append(_prompt)
                cls.output_data["completion"].append(_completion)
            if cls.input_data["prompt"]:
                cls._update_message(cls)
                prompt = cls.input_data["prompt"].pop(_POP_INDEX)
                completion = cls.input_data["completion"].pop(_POP_INDEX)
                completion = completion if fn is None or completion else fn(prompt)
                return prompt, completion
            else:
                cls._done_message()
                return None, None

        # UI Config
        prompt, completion = next_input(None, None)
        input_prompt = gradio.Textbox(value=prompt, label="prompt")
        input_completion = gradio.Image(
            value=completion, label="completion", height=400
        )
        inputs = [input_prompt, input_completion]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            flagging_options=_SUBMIT_OPTIONS,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_image_description(
        cls,
        images: Optional[List[Union[np.array, PIL.Image.Image, str]]] = None,
        descriptions: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image description tasks.

        Parameters:
            images (List[Union[np.array, PIL.Image.Image, str]]): List of images to annotate.
            descriptions (Optional[List[str]]): List of descriptions to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]]): Prediction function to apply to the image before annotating.
                it should take an `np.array` or `PIL.Image.Image` and return `str`.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # IO Config
        cls.task = "image-description"
        cls.input_columns = ["image", "description"]
        cls.output_columns = ["image", "description"]
        cls.input_data = {"image": images or [], "description": descriptions or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["image"])

        # Input validation
        if cls.input_data["prompt"] and cls.input_data["completion"]:
            if len(cls.input_data["prompt"]) != len(cls.input_data["completion"]):
                raise ValueError(
                    "Source and target must be of the same length. You can add empty strings to match the lengths."
                )

        # Process function
        def next_input(_image, _description):
            if _image:
                cls.output_data["image"].append(_image)
                cls.output_data["description"].append(_description)
            if cls.input_data["image"]:
                cls._update_message(cls)
                image = cls.input_data["image"].pop(_POP_INDEX)
                description = cls.input_data["description"].pop(_POP_INDEX)
                description = description if fn is None or description else fn(image)
                return image, description
            else:
                cls._done_message()
                return None, ""

        # UI Config
        image, description = next_input(None, None)
        inputs = gradio.Image(value=image, label="image", height=400)
        outputs = gradio.Textbox(value=description, label="text")
        inputs = [inputs, outputs]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=_SUBMIT_OPTIONS,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_image_generation_preference(
        cls,
        prompts: Optional[List[str]] = None,
        completions_a: Optional[List[Union[np.array, PIL.Image.Image, str]]] = None,
        completions_b: Optional[List[Union[np.array, PIL.Image.Image, str]]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image generation preference tasks.

        Parameters:
            prompts (List[str]): List of prompts to annotate.
            completions_a (List[Union[np.array, PIL.Image.Image, str]]): List of completions to annotate for option A.
            completions_b (List[Union[np.array, PIL.Image.Image, str]]): List of completions to annotate for option B.
            fn (Optional[Union["Pipeline", callable]]): NotImplementedError. Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.
        """
        # IO Config
        cls.task = "image-generation-preference"
        cls.input_columns = ["prompt", "completion_a", "completion_b"]
        cls.output_columns = ["prompt", "chosen", "rejected", "flag"]
        cls.input_data = {
            "prompt": prompts or [],
            "completion_a": completions_a or [],
            "completion_b": completions_b or [],
        }
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["prompt"])

        # Input validation
        if (
            cls.input_data["prompt"]
            and cls.input_data["completion_a"]
            and cls.input_data["completion_b"]
        ):
            if (
                len(cls.input_data["prompt"])
                != len(cls.input_data["completion_a"])
                != len(cls.input_data["completion_b"])
            ):
                raise ValueError("Prompts and completions must be of the same length")

        # Process function
        def next_input(_prompt, _completion_a, _completion_b):
            if _prompt:
                cls.output_data["prompt"].append(_prompt)
                if cls.output_data["flag"][-1] == "👆 A is better":
                    cls.output_data["chosen"].append(_completion_a)
                    cls.output_data["rejected"].append(_completion_b)
                else:
                    cls.output_data["chosen"].append(_completion_b)
                    cls.output_data["rejected"].append(_completion_a)
            if cls.input_data["prompt"]:
                cls._update_message(cls)
                prompt = cls.input_data["prompt"].pop(_POP_INDEX)
                completion_a = cls.input_data["completion_a"].pop(_POP_INDEX)
                completion_b = cls.input_data["completion_b"].pop(_POP_INDEX)
                completion_a = (
                    completion_a if fn is None or completion_a else fn(prompt)
                )
                completion_b = (
                    completion_b if fn is None or completion_b else fn(prompt)
                )
                return prompt, completion_a, completion_b
            else:
                cls._done_message()
                return None, None, None

        # UI Config
        prompt, completion_a, completion_b = next_input(None, None, None)
        input_prompt = gradio.Textbox(value=prompt, label="prompt")
        input_completion_a = gradio.Image(
            value=completion_a, label="👆 completion A", height=400
        )
        input_completion_b = gradio.Image(
            value=completion_b, label="👇 completion B", height=400
        )
        inputs = [input_prompt, input_completion_a, input_completion_b]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            flagging_options=_PREFERENCE_OPTIONS,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_image_question_answering(
        cls,
        images: List[Union[np.array, PIL.Image.Image, str]] = None,
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image question answering tasks.

        Parameters:
            images (List[Union[np.array, PIL.Image.Image, str]]): List of images to annotate.
            questions (Optional[List[str]]): List of questions to annotate. Defaults to None.
            answers (Optional[List[str]]): List of answers to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]]): NotImplementedError. Defaults to None.
            dataset_name (Optional[str]): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str]): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool]): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        if fn:
            raise NotImplementedError(
                "Prediction function is not supported for this task."
            )
        # IO Config
        cls.task = "image-question-answering"
        cls.input_columns = ["image", "question", "answer"]
        cls.output_columns = ["image", "question", "answer"]
        cls.input_data = {"image": images or []}
        cls.output_data = {col: [] for col in cls.output_columns}
        cls.start = len(cls.input_data["image"])

        # Input validation
        if cls.input_data["image"]:
            max_length = max(len(img) for img in cls.input_data["image"])
            cls.input_data["question"] = [
                question.ljust(max_length) for question in cls.input_data["question"]
            ]
            cls.input_data["answer"] = [
                answer.ljust(max_length) for answer in cls.input_data["answer"]
            ]

        # Process function
        def next_input(_image, _question, _answer):
            if _image:
                cls.output_data["image"].append(_image)
                cls.output_data["question"].append(_question)
                cls.output_data["answer"].append(_answer)
            if cls.input_data["image"]:
                cls._update_message(cls)
                image = cls.input_data["image"].pop(_POP_INDEX)
                question = cls.input_data["question"].pop(_POP_INDEX)
                answer = cls.input_data["answer"].pop(_POP_INDEX)
                return image, question, answer
            else:
                cls._done_message()
                return None, "", ""

        # UI Config
        image, question, answer = next_input(None, None, None)
        inputs = [
            gradio.Image(value=image, label="image", height=400),
            gradio.Textbox(value=question, label="question"),
            gradio.Textbox(value=answer, label="answer"),
        ]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=_SUBMIT_OPTIONS,
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @override
    def attach_flagging_events(
        self,
        flag_btns: list[Button, gradio.CheckboxGroup] | None,
        _clear_btn: ClearButton,
        _submit_event: Dependency,
    ) -> None:
        """Override the attach_flagging_events method to attach the flagging events."""
        # before the flaffing because otherwise input is reset
        self.attach_submit_events(_submit_btn=_clear_btn, _stop_btn=None)
        for btn in flag_btns:

            def add_label():
                if "flag" in self.output_data:
                    self.output_data["flag"].append(str(btn.value))

            btn.click(fn=add_label)
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
        """Override the render_flag_btns method to return the flagging buttons with labels."""
        self.flagging_btns = [
            Button(label, variant="primary") for label, _ in self.flagging_options
        ]
        return self.flagging_btns

    def _update_message(self) -> None:
        """Print the progress of the annotation."""
        key = list(self.input_data.keys())[0]
        items = self.input_data[key]
        start = len(self.input_data[key]) + len(self.output_data[key])
        gradio.Info(f"{(len(items) / start) * 100:.2f}% done {len(items)} left.")

    @staticmethod
    def _done_message() -> None:
        """Print the done message."""
        gradio.Info(_MESSAGE_DONE_ANNOTATING)

    @staticmethod
    def _convert_to_tokens(text: str) -> List[tuple[str, None]]:
        """Convert a string to a list of tokens for TextHighligh."""
        return [(char, None) for char in text]

    @staticmethod
    def _convert_to_chat_message(
        messages: Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]],
        with_turn=False,
        last_role=None,
    ) -> List[List[gradio.ChatMessage]]:
        """
        Convert a list of chat messages to a list of gradio.ChatMessage.

        Parameters:
            messages (Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]): List of chat messages.
            with_turn (bool): Whether to add turn information. Defaults to False.
            last_role ([type]): Last role. Defaults to None.

        Returns:
            List[List[gradio.ChatMessage]]: List of chat messages.
        """
        if not messages:
            return []
        if not isinstance(messages[0][0], gradio.ChatMessage):
            messages = [
                [gradio.ChatMessage(**msg) for msg in prompt] for prompt in messages
            ]
        if with_turn:
            messages = [
                [
                    gradio.ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        metadata={
                            "title": f"Turn {idx} - role: {msg.role} - length {len(msg.content)}"
                        },
                    )
                    for idx, msg in enumerate(prompt)
                ]
                for prompt in messages
            ]
        if last_role is not None:
            for prompt in messages:
                assert (
                    prompt[-1].role == last_role
                ), f"Last message role should be {last_role}."
        return messages

    @staticmethod
    def _validate_preference(fn, prompts, completions_a, completions_b):
        """
        Validate the inputs for preference tasks.

        Parameters:
            fn: Prediction function.
            prompts: List of prompts.
            completions_a: List of completions for option A.
            completions_b: List of completions for option B.

        Returns:
            Tuple: Tuple of prompts, completions_a, completions_b.
        """
        if fn is not None and (completions_a is not None and completions_b is not None):
            raise ValueError("fn should be None when completions are provided.")
        if completions_a is None:
            completions_a = ["" for _ in range(len(prompts))]
        if completions_b is None:
            completions_b = ["" for _ in range(len(prompts))]
        if any(
            [len(prompts) != len(completions_a), len(prompts) != len(completions_b)]
        ):
            raise ValueError("Prompts and completions must be of the same length")
        return prompts, completions_a, completions_b

    def shuffle_data(self):
        if not self.input_data.values():
            return self.input_data

        # Get the length of the first list in the dictionary
        first_key = next(iter(self.input_data))
        length = len(self.input_data[first_key])

        # Check if all lists have the same length
        if not all(len(lst) == length for lst in self.input_data.values()):
            raise ValueError("All input lists must have the same length")

        # Create a list of indices and shuffle it
        indices = list(range(length))
        random.shuffle(indices)

        # Reorder each list based on the shuffled indices
        gradio.Info("Data shuffled.")
        self.input_data = {
            key: [lst[i] for i in indices] for key, lst in self.input_data.items()
        }
        return pd.DataFrame.from_dict(self.input_data).head(100)
