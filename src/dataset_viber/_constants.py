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

TASK_MAPPING = {
    "text-classification": {
        "input_columns": ["text", "suggestion"],
        "output_columns": ["text", "label"],
        "fn_model_output": "List[Dict[str, Union[str, float]]]",
        "fn_next_input_output": "Tuple[str, str]",
    },
    "text-classification-multi-label": {
        "input_columns": ["text", "suggestion"],
        "output_columns": ["text", "label"],
        "fn_model_output": "List[Dict[str, Union[str, float]]]",
        "fn_next_input_output": "Tuple[str, List[str]]",
    },
    "token-classification": {
        "input_columns": ["text"],
        "output_columns": ["text", "spans"],
        "fn_model_output": "List[Tuple[str, str]]",
        "fn_next_input_output": "Tuple[str, List[Tuple[str, str]]]",
    },
    "question-answering": {
        "input_columns": ["question", "context"],
        "output_columns": ["question", "context"],
        "fn_model_output": "List[Tuple[str, str]]",
        "fn_next_input_output": "Tuple[str, Union[str, List[Tuple[str, str]]]]",
    },
    "text-generation": {
        "input_columns": ["prompt", "completion"],
        "output_columns": ["prompt", "completion"],
        "fn_model_output": "str",
        "fn_next_input_output": "Tuple[str, str]",
    },
    "text-generation-preference": {
        "input_columns": ["prompt", "completion_a", "completion_b"],
        "output_columns": ["prompt", "completion_a", "completion_b", "flag"],
        "fn_model_output": "str",
        "fn_next_input_output": "Tuple[str, str, str]",
    },
    "chat-classification": {
        "input_columns": ["prompt", "suggestion"],
        "output_columns": ["prompt", "label"],
        "fn_model_output": "List[Dict[str, Union[str, float]]]",
        "fn_next_input_output": "Tuple[List[gradio.ChatMessage], str]",
    },
    "chat-classification-multi-label": {
        "input_columns": ["prompt", "suggestion"],
        "output_columns": ["prompt", "label"],
        "fn_model_output": "List[Dict[str, Union[str, float]]]",
        "fn_next_input_output": "Tuple[List[gradio.ChatMessage], List[str]]",
    },
    "chat-generation": {
        "input_columns": ["prompt", "completion"],
        "output_columns": ["prompt", "completion"],
        "fn_model_output": "str",
        "fn_next_input_output": "Tuple[List[gradio.ChatMessage], str]",
    },
    "chat-generation-preference": {
        "input_columns": ["prompt", "completion_a", "completion_b"],
        "output_columns": ["prompt", "completion_a", "completion_b", "flag"],
        "fn_model_output": "str",
        "fn_next_input_output": "Tuple[List[gradio.ChatMessage], str, str]",
    },
    "image-classification": {
        "input_columns": ["image", "suggestion"],
        "output_columns": ["image", "label"],
        "fn_model_output": "Union[str, List[Dict[str, Union[str, float]]]]",
        "fn_next_input_output": "Tuple[PIL.Image.Image, str]",
    },
    "image-classification-multi-label": {
        "input_columns": ["image", "suggestion"],
        "output_columns": ["image", "label"],
        "fn_model_output": "List[Dict[str, Union[str, float]]]",
        "fn_next_input_output": "Tuple[PIL.Image.Image, List[str]]",
    },
    "image-generation": {
        "input_columns": ["prompt", "completion"],
        "output_columns": ["prompt", "completion"],
        "fn_model_output": "Union[np.array, PIL.Image.Image, str]",
        "fn_next_input_output": "Tuple[str, PIL.Image.Image]",
    },
    "image-description": {
        "input_columns": ["image", "description"],
        "output_columns": ["image", "description"],
        "fn_model_output": "str",
        "fn_next_input_output": "Tuple[PIL.Image.Image, str]",
    },
    "image-generation-preference": {
        "input_columns": ["prompt", "completion_a", "completion_b"],
        "output_columns": ["prompt", "completion_a", "completion_b", "flag"],
        "fn_model_output": "PIL.Image.Image",
        "fn_next_input_output": "Tuple[str, PIL.Image.Image, PIL.Image.Image]",
    },
    "image-question-answering": {
        "input_columns": ["image", "question", "answer"],
        "output_columns": ["image", "question", "answer"],
        "fn_model_output": "str",
        "fn_next_input_output": "Tuple[PIL.Image.Image, str, str]",
    },
}
