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

from typing import Dict, List, Tuple, Union

import argilla as rg
import gradio
import numpy as np
import PIL.Image


class MockClient:
    def __init__(self):
        self.api = MockApi()
        self.workspaces = lambda _: rg.Workspace(name="dataset-viber", client=self)


class MockApi:
    def __init__(self):
        self.fields = ""
        self.datasets = ""
        self.questions = ""
        self.records = ""


DEFAULT_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-xs"
COLORS = [
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
FAKE_UUID = "00000000-0000-0000-0000-000000000000"
DEFAULT_DATASET_CONFIG = {
    "id": FAKE_UUID,
    "inserted_at": "2024-07-30T18:54:05.550199",
    "updated_at": "2024-07-30T18:54:05.748298",
    "name": "dataset-viber",
    "status": "ready",
    "guidelines": None,
    "allow_extra_metadata": True,
    "distribution": {"strategy": "overlap", "min_submitted": 1},
    "workspace_id": FAKE_UUID,
    "last_activity_at": FAKE_UUID,
}

TASK_MAPPING = {
    "text-classification": {
        "input_columns": ["text", "suggestion"],
        "output_columns": ["text", "label"],
        "fn_model_output": List[Dict[str, Union[str, float]]],
        "fn_next_input_output": Tuple[str, str],
        "components": [gradio.Textbox, gradio.Radio],
        "autotrain": {
            "hub": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/text_classification/hub_dataset.yml",
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/text_classification/local_dataset.yml",
        },
    },
    "text-classification-multi-label": {
        "input_columns": ["text", "suggestion"],
        "output_columns": ["text", "label"],
        "fn_model_output": List[Dict[str, Union[str, float]]],
        "fn_next_input_output": Tuple[str, List[str]],
        "components": [gradio.Textbox, gradio.CheckboxGroup],
        "autotrain": {
            "hub": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/text_classification/hub_dataset.yml",
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/text_classification/local_dataset.yml",
        },
    },
    "token-classification": {
        "input_columns": ["text"],
        "output_columns": ["text", "spans"],
        "fn_model_output": List[Tuple[str, str]],
        "fn_next_input_output": Tuple[str, List[Tuple[str, str]]],
        "components": [gradio.Textbox, gradio.HighlightedText],
        "autotrain": {
            "hub": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/token_classification/hub_dataset.yml",
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/token_classification/local_dataset.yml",
        },
    },
    "question-answering": {
        "input_columns": ["question", "context"],
        "output_columns": ["question", "context"],
        "fn_model_output": List[Tuple[str, str]],
        "fn_next_input_output": Tuple[str, Union[str, List[Tuple[str, str]]]],
        "components": [gradio.Textbox, gradio.HighlightedText],
        "autotrain": {
            "hub": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/extractive_question_answering/hub_dataset.yml",
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/extractive_question_answering/local_dataset.yml",
        },
    },
    "text-generation": {
        "input_columns": ["prompt", "completion"],
        "output_columns": ["prompt", "completion"],
        "fn_model_output": str,
        "fn_next_input_output": Tuple[str, str],
        "components": [gradio.Textbox, gradio.Textbox],
        "autotrain": {
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/chat_generation/local.yml"
        },
    },
    "text-generation-preference": {
        "input_columns": ["prompt", "completion_a", "completion_b"],
        "output_columns": ["prompt", "completion_a", "completion_b", "flag"],
        "fn_model_output": str,
        "fn_next_input_output": Tuple[str, str, str],
        "components": [gradio.Textbox, gradio.Textbox, gradio.Textbox],
        "autotrain": {
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/llm_finetuning/llama3-8b-orpo.yml"
        },
    },
    "chat-classification": {
        "input_columns": ["prompt", "suggestion"],
        "output_columns": ["prompt", "label"],
        "fn_model_output": List[Dict[str, Union[str, float]]],
        "fn_next_input_output": Tuple[List[gradio.ChatMessage], str],
        "components": [gradio.Chatbot, gradio.Radio],
    },
    "chat-classification-multi-label": {
        "input_columns": ["prompt", "suggestion"],
        "output_columns": ["prompt", "label"],
        "fn_model_output": List[Dict[str, Union[str, float]]],
        "fn_next_input_output": Tuple[List[gradio.ChatMessage], List[str]],
        "components": [gradio.Chatbot, gradio.CheckboxGroup],
    },
    "chat-generation": {
        "input_columns": ["prompt", "completion"],
        "output_columns": ["prompt", "completion"],
        "fn_model_output": str,
        "fn_next_input_output": Tuple[List[gradio.ChatMessage], str],
        "components": [gradio.Chatbot, gradio.Textbox],
        "autotrain": {
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/chat_generation/local.yml"
        },
    },
    "chat-generation-preference": {
        "input_columns": ["prompt", "completion_a", "completion_b"],
        "output_columns": ["prompt", "completion_a", "completion_b", "flag"],
        "fn_model_output": str,
        "fn_next_input_output": Tuple[List[gradio.ChatMessage], str, str],
        "components": [gradio.Chatbot, gradio.Textbox, gradio.Textbox],
        "autotrain": {
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/llm_finetuning/llama3-8b-orpo.yml"
        },
    },
    "image-classification": {
        "input_columns": ["image", "suggestion"],
        "output_columns": ["image", "label"],
        "fn_model_output": Union[str, List[Dict[str, Union[str, float]]]],
        "fn_next_input_output": Tuple[PIL.Image.Image, str],
        "components": [gradio.Image, gradio.Radio],
        "autotrain": {
            "hub": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/image_classification/hub_dataset.yml",
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/image_classification/local.yml",
        },
    },
    "image-classification-multi-label": {
        "input_columns": ["image", "suggestion"],
        "output_columns": ["image", "label"],
        "fn_model_output": List[Dict[str, Union[str, float]]],
        "fn_next_input_output": Tuple[PIL.Image.Image, List[str]],
        "components": [gradio.Image, gradio.CheckboxGroup],
        "autotrain": {
            "hub": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/image_classification/hub_dataset.yml",
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/image_classification/local.yml",
        },
    },
    "image-generation": {
        "input_columns": ["prompt", "completion"],
        "output_columns": ["prompt", "completion"],
        "fn_model_output": Union[np.ndarray, PIL.Image.Image, str],
        "fn_next_input_output": Tuple[str, PIL.Image.Image],
        "components": [gradio.Textbox, gradio.Image],
    },
    "image-description": {
        "input_columns": ["image", "description"],
        "output_columns": ["image", "description"],
        "fn_model_output": str,
        "fn_next_input_output": Tuple[PIL.Image.Image, str],
        "components": [gradio.Image, gradio.Textbox],
    },
    "image-generation-preference": {
        "input_columns": ["prompt", "completion_a", "completion_b"],
        "output_columns": ["prompt", "completion_a", "completion_b", "flag"],
        "fn_model_output": PIL.Image.Image,
        "fn_next_input_output": Tuple[str, PIL.Image.Image, PIL.Image.Image],
        "components": [gradio.Textbox, gradio.Image, gradio.Image],
    },
    "image-question-answering": {
        "input_columns": ["image", "question", "answer"],
        "output_columns": ["image", "question", "answer"],
        "fn_model_output": str,
        "fn_next_input_output": Tuple[PIL.Image.Image, str, str],
        "components": [gradio.Image, gradio.Textbox, gradio.Textbox],
        "autotrain": {
            "local": "https://github.com/huggingface/autotrain-advanced/blob/main/configs/vlm/paligemma_vqa.yml"
        },
    },
}
