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

import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import gradio
import numpy as np
import PIL
from gradio.components import (
    Button,
    ClearButton,
)
from gradio.events import Dependency
from gradio.flagging import FlagMethod

from dataset_viber._constants import COLORS
from dataset_viber._gradio.collector import CollectorInterface

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

_POP_INDEX = 0
_MESSAGE_DONE_ANNOTATING = "No data left to annotate."
_HIGHLIGHT_TEXT_KWARGS = {
    "interactive": True,
    "show_legend": False,
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


class AnnotatorInterFace(CollectorInterface):
    @classmethod
    def for_text_classification(
        cls,
        texts: List[str],
        labels: List[str],
        *,
        multi_label: Optional[bool] = False,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for text classification tasks.

        Args:
            texts (List[str]): List of texts to annotate.
            labels (List[str]): List of labels to choose from.
            multi_label (Optional[bool], optional): Whether to allow multiple labels. Defaults to False.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the text before annotating.
                Expecting it takes a `str` and returns [{"label": str, "score": float}].
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Process function
        start = len(texts)

        def next_input(_text, _label):
            if texts:
                cls._update_message(texts, start)
                text = texts.pop(_POP_INDEX)
                label = [] if fn is None else fn(text)
                if multi_label:
                    label = [lab["label"] for lab in label if lab["score"] > 0.5]
                    return (text, label)
                elif not multi_label and fn is not None:
                    return (text, label[0]["label"])
                else:
                    return text
            else:
                cls._done_message()
                return ("", []) if multi_label or fn is not None else ""

        if multi_label or fn is not None:
            text, label = next_input(None, None)
            if multi_label:
                check_box_group = gradio.CheckboxGroup(
                    labels, value=label, label="label"
                )
            else:
                check_box_group = gradio.Radio(labels, value=label, label="label")
        else:
            text = next_input(None, None)

        # UI Config
        inputs = gradio.Textbox(value=text, label="text")
        if multi_label or fn is not None:
            inputs = [inputs, check_box_group]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=(
                _SUBMIT_OPTIONS
                if multi_label or fn is not None
                else [(lab, lab) for lab in labels]
            ),
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
        texts: List[str],
        labels: List[str],
        *,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for token classification tasks.

        Args:
            texts (List[str]): List of texts to annotate.
            labels (List[str]): List of labels to choose from.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the text before annotating.
                Expecting it takes a `str` and returns List[Tuple[str, str]] [("text","label")].
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Process function
        start = len(texts)

        def next_input(_spans):
            if texts:
                cls._update_message(texts, start)
                text = texts.pop(_POP_INDEX)
                text = cls._convert_to_tokens(text) if fn is None else fn(text)
                return text
            else:
                cls._done_message()
                return cls._convert_to_tokens(" ")

        # UI Config
        if isinstance(labels, list):
            labels = {label: color for label, color in zip(labels, COLORS)}
        text = next_input(None)
        inputs = [
            gradio.HighlightedText(
                value=text,
                color_map=labels,
                label="spans",
                **_HIGHLIGHT_TEXT_KWARGS,
            )
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
        questions: List[str],
        contexts: List[str],
        *,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for question answering tasks.

        Args:
            questions (List[str]): List of questions to annotate.
            contexts (List[str]): List of contexts to annotate.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the context before annotating.
                Expecting it takes a `str` and returns [{"label": str, "score": float}].
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        start = len(questions)
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must be of the same length.")

        # Process function
        def next_input(_question, _context):
            if questions:
                cls._update_message(questions, start)
                question = questions.pop(_POP_INDEX)
                context = contexts.pop(_POP_INDEX)
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
        prompts: List[str],
        *,
        completions: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for text generation tasks.

        Args:
            prompts (List[str]): List of prompts to annotate.
            completions (Optional[List[str]], optional): List of completions to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the prompt before annotating.
                Expecting it takes a `str` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        if completions is None:
            completions = ["" for _ in range(len(prompts))]
        elif fn is not None:
            raise ValueError("fn should be None when completions are provided.")
        if len(prompts) != len(completions):
            raise ValueError(
                "Source and target must be of the same length. You can add empty strings to match the lengths."
            )

        # Process function
        start = len(prompts)

        def next_input(_prompt, _completion):
            if prompts:
                cls._update_message(prompts, start)
                prompt = prompts.pop(_POP_INDEX)
                completion = completions.pop(_POP_INDEX) if fn is None else fn(prompt)
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
        prompts: List[str],
        *,
        completions_a: Optional[List[str]] = None,
        completions_b: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for text generation preference tasks.

        Args:
            prompts (List[str]): List of prompts to annotate.
            completions_a (Optional[List[str]], optional): List of completions to annotate for option A. Defaults to None.
            completions_b (Optional[List[str]], optional): List of completions to annotate for option B. Defaults to None.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the prompt before annotating.
                Expecting it takes a `str` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        start = len(prompts)
        prompts, completions_a, completions_b = cls._validate_preference(
            cls, fn, prompts, completions_a, completions_b
        )

        # Process function
        def next_input(_prompt, _completion_a, _completion_b):
            if prompts:
                cls._update_message(prompts, start)
                prompt = prompts.pop(_POP_INDEX)
                completion_a = completions_a.pop(_POP_INDEX)
                completion_a = (
                    completion_a if fn is None or completion_a != "" else fn(prompt)
                )
                completion_b = completions_b.pop(_POP_INDEX)
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
        prompts: List[List[Dict[str, str]]],
        labels: List[str],
        *,
        multi_label: Optional[bool] = False,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for chat classification tasks.

        Args:
            prompts (List[List[Dict[str, str]]): List of chat messages to annotate.
            labels (List[str]): List of labels to choose from.
            multi_label (Optional[bool], optional): Whether to allow multiple labels. Defaults to False.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the chat messages before annotating.
                Expecting it takes a `str` and returns [{"label": str, "score": float}].
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        start = len(prompts)
        prompts = cls._convert_to_chat_message(prompts)

        # Process function
        def next_input(_prompt, _label):
            if prompts:
                cls._update_message(prompts, start)
                prompt = prompts.pop(_POP_INDEX)
                label = (
                    [] if fn is None else fn("\n".join([msg.content for msg in prompt]))
                )
                if multi_label:
                    return (
                        prompt,
                        [lab["label"] for lab in label if label["score"] > 0.5],
                    )
                elif not multi_label and fn is not None:
                    return (prompt, label[0]["label"])
                else:
                    return prompt
            else:
                cls._done_message()
                return ([], []) if multi_label or fn is not None else []

        # UI Config
        if multi_label:
            labels = [(label, label) for label in labels]
        if multi_label or fn is not None:
            prompt, label = next_input(None, None)
            if multi_label:
                check_box_group = gradio.CheckboxGroup(
                    labels, value=label, label="label"
                )
            else:
                check_box_group = gradio.Radio(labels, value=label, label="label")
        else:
            prompt = next_input(None, None)
        inputs = gradio.Chatbot(
            value=prompt,
            **_CHATBOT_KWARGS,
        )
        if multi_label or fn is not None:
            inputs = [inputs, check_box_group]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=(
                _SUBMIT_OPTIONS if multi_label else [(lab, lab) for lab in labels]
            ),
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_chat_classification_per_message(
        cls,
        prompts: List[List[Dict[str, str]]],
        labels: List[str],
        *,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        if fn is not None:
            raise NotImplementedError(
                "fn is not supported for chat message classification."
            )
        # Input validation
        start = len(prompts)
        prompts = cls._convert_to_chat_message(prompts, with_turn=True)

        # Process function
        def next_input(_prompt, *args):
            if prompts:
                cls._update_message(prompts, start)
                prompt = prompts.pop(_POP_INDEX)
                return prompt, *[""] * len(labels)
            else:
                cls._done_message()
                return "", *[""] * len(labels)

        # UI Config
        prompt, *args = next_input(None, None)
        chatbot = gradio.Chatbot(
            value=prompt,
            **_CHATBOT_KWARGS,
        )
        inputs = [
            gradio.Dropdown(choices=range(100), label=label, multiselect=True)
            for label in labels
        ]
        inputs = [chatbot] + inputs
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
    def for_chat_generation(
        cls,
        prompts: Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]],
        *,
        completions: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for chat generation tasks.

        Args:
            prompts (Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]): List of chat messages to annotate.
            completions (Optional[List[str]], optional): List of completions to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the chat messages before annotating.
                Expecting it takes a `List[gradio.ChatMessage]` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        start = len(prompts)
        if completions is None:
            completions = ["" for _ in range(len(prompts))]
        if len(prompts) != len(completions):
            raise ValueError(
                "Source and target must be of the same length. You can add empty strings to match the lengths."
            )
        prompts = cls._convert_to_chat_message(prompts)

        # Process function
        def next_input(_prompt, _completion):
            def _last_is_user(_prompt):
                return _prompt[-1].role == "user"

            if prompts:
                if _prompt and False:
                    _prompt.append(
                        gradio.ChatMessage(
                            role="assistant" if _last_is_user(_prompt) else "user",
                            content=_completion,
                        )
                    )
                    prompt = _prompt
                    completion = ""
                    if _last_is_user(prompt):
                        completion = "" if fn is None else fn(prompt)
                else:
                    cls._update_message(prompts, start)
                    prompt = prompts.pop(_POP_INDEX)
                    completion = completions.pop(_POP_INDEX)
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
        prompts: Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]],
        *,
        completions_a: Optional[List[str]],
        completions_b: Optional[List[str]],
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for chat generation preference tasks.

        Args:
            prompts (Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]): List of chat messages to annotate.
            completions_a (Optional[List[str]]): List of completions to annotate for option A.
            completions_b (Optional[List[str]]): List of completions to annotate for option B.
            fn (Optional[Union["Pipeline", callable]], optional): Prediction function to apply to the chat messages before annotating.
                Expecting it takes a `List[gradio.ChatMessage]` and returns `str`.
                Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        start = len(prompts)
        prompts, completions_a, completions_b = cls._validate_preference(
            fn, prompts, completions_a, completions_b
        )
        prompts = cls._convert_to_chat_message(prompts)

        # Process function
        def next_input(_prompt, _completion_a, _completion_b):
            if prompts:
                cls._update_message(prompts, start)
                prompt = prompts.pop(_POP_INDEX)
                completion_a = completions_a.pop(_POP_INDEX)
                completion_a = (
                    completion_a if fn is None or completion_a else fn(prompt)
                )
                completion_b = completions_b.pop(_POP_INDEX)
                completion_b = (
                    completion_b if fn is None or completion_b else fn(prompt)
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
    def for_image_generation_preference(
        cls,
        prompts: List[str],
        completions_a: List[Union[np.array, PIL.Image.Image, str]],
        completions_b: List[Union[np.array, PIL.Image.Image, str]],
        *,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image generation preference tasks.

        Args:
            prompts (List[str]): List of prompts to annotate.
            completions_a (List[Union[np.array, PIL.Image.Image, str]]): List of completions to annotate for option A.
            completions_b (List[Union[np.array, PIL.Image.Image, str]]): List of completions to annotate for option B.
            fn (Optional[Union["Pipeline", callable]], optional): NotImplementedError. Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.
        """
        # Input validation
        start = len(prompts)
        if fn is not None:
            raise NotImplementedError(
                "fn is not supported for image generation preference."
            )
        if len(prompts) != len(completions_a) != len(completions_b):
            raise ValueError("Prompts and completions must be of the same length")

        # Process function
        def next_input(_prompt, _completion_a, _completion_b):
            if prompts:
                cls._update_message(prompts, start)
                return (
                    prompts.pop(_POP_INDEX),
                    completions_a.pop(_POP_INDEX),
                    completions_b.pop(_POP_INDEX),
                )
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
    def for_image_classification(
        cls,
        images: List[Union[np.array, PIL.Image.Image, str]],
        labels: List[str],
        *,
        multi_label: Optional[bool] = False,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image classification tasks.

        Args:
            images (List[Union[np.array, PIL.Image.Image, str]]): List of images to annotate.
            labels (List[str]): List of labels to choose from.
            multi_label (Optional[bool], optional): Whether to allow multiple labels. Defaults to False.
            fn (Optional[Union["Pipeline", callable]], optional): NotImplementedError. Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        start = len(images)
        if fn is not None:
            raise NotImplementedError("fn is not supported for image classification.")

        # Process function
        def next_input(_image, _label):
            if images:
                cls._update_message(images, start)
                image = images.pop(_POP_INDEX)
                return (image, []) if multi_label else image
            else:
                cls._done_message()
                return (None, []) if multi_label else None

        # UI Config
        labels = [(label, label) for label in labels]
        inputs = gradio.Image(value=images.pop(_POP_INDEX), label="image", height=400)
        if multi_label:
            inputs = [inputs, gradio.CheckboxGroup(labels, label="label")]
        return cls(
            fn=next_input,
            inputs=inputs,
            outputs=inputs,
            flagging_options=(
                _SUBMIT_OPTIONS if multi_label else [(lab, lab) for lab in labels]
            ),
            allow_flagging="manual",
            submit_btn=_SUBMIT_BTN,
            clear_btn=_CLEAR_BTN,
            dataset_name=dataset_name,
            hf_token=hf_token,
            private=private,
        )

    @classmethod
    def for_image_description(
        cls,
        images: List[Union[np.array, PIL.Image.Image, str]],
        *,
        descriptions: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image description tasks.

        Args:
            images (List[Union[np.array, PIL.Image.Image, str]]): List of images to annotate.
            descriptions (Optional[List[str]], optional): List of descriptions to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]], optional): NotImplementedError. Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        # Input validation
        start = len(images)
        if fn is not None:
            raise NotImplementedError("fn is not supported for image description.")
        if descriptions is None:
            descriptions = ["" for _ in range(len(images))]
        else:
            assert len(descriptions) == len(images)

        # Process function
        def next_input(_image, _description):
            if images:
                cls._update_message(images, start)
                img = images.pop(_POP_INDEX)
                description = descriptions.pop(_POP_INDEX)
                return img, description
            else:
                cls._done_message()
                return None, ""

        # UI Config
        inputs = gradio.Image(value=images.pop(_POP_INDEX), label="image", height=400)
        outputs = gradio.Textbox(value=descriptions.pop(_POP_INDEX), label="text")
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
    def for_image_question_answering(
        cls,
        images: List[Union[np.array, PIL.Image.Image, str]],
        *,
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        fn: Optional[Union["Pipeline", callable]] = None,
        dataset_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: Optional[bool] = False,
    ) -> "AnnotatorInterFace":
        """
        Annotator Interface for image question answering tasks.

        Args:
            images (List[Union[np.array, PIL.Image.Image, str]]): List of images to annotate.
            questions (Optional[List[str]], optional): List of questions to annotate. Defaults to None.
            answers (Optional[List[str]], optional): List of answers to annotate. Defaults to None.
            fn (Optional[Union["Pipeline", callable]], optional): NotImplementedError. Defaults to None.
            dataset_name (Optional[str], optional): Name of the dataset to save the annotations. Defaults to None.
            hf_token (Optional[str], optional): Hugging Face API token to save the annotations. Defaults to None.
            private (Optional[bool], optional): Whether to save the annotations as private. Defaults to False.

        Returns:
            AnnotatorInterFace: An instance of AnnotatorInterFace
        """
        start = len(images)
        if fn is not None:
            raise NotImplementedError(
                "fn is not supported for image question answering."
            )
        if questions is None:
            questions = ["" for _ in range(start)]
        else:
            assert len(questions) == start
        if answers is None:
            answers = ["" for _ in range(start)]
        else:
            assert len(answers) == start

        # Process function
        def next_input(_image, _question, _answer):
            if images:
                cls._update_message(images, start)
                img = images.pop(_POP_INDEX)
                question = questions.pop(_POP_INDEX)
                answer = answers.pop(_POP_INDEX)
                return img, question, answer
            else:
                cls._done_message()
                return None, "", ""

        # UI Config
        inputs = gradio.Image(value=images.pop(_POP_INDEX), label="image", height=400)
        outputs = [
            gradio.Textbox(value=questions.pop(_POP_INDEX), label="question"),
            gradio.Textbox(value=answers.pop(_POP_INDEX), label="answer"),
        ]
        inputs = [inputs, *outputs]
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
        return [Button(label, variant="primary") for label, _ in self.flagging_options]

    @staticmethod
    def _update_message(items: list, start: int) -> None:
        """Print the progress of the annotation."""
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

        Args:
            messages (Union[List[List[Dict[str, str]]], List[List[gradio.ChatMessage]]): List of chat messages.
            with_turn (bool, optional): Whether to add turn information. Defaults to False.
            last_role ([type], optional): Last role. Defaults to None.

        Returns:
            List[List[gradio.ChatMessage]]: List of chat messages.
        """
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

        Args:
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
