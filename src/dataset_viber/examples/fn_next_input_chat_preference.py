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

import os
import random

from datasets import load_dataset
from huggingface_hub import InferenceClient

from dataset_viber import AnnotatorInterFace

# https://huggingface.co/models?inference=warm&pipeline_tag=text-generation&sort=trending
MODEL_IDS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]
CLIENTS = [
    InferenceClient(model_id, token=os.environ["HF_TOKEN"]) for model_id in MODEL_IDS
]

dataset = load_dataset("argilla/magpie-ultra-v0.1", split="train")


def _get_response(messages):
    client = random.choice(CLIENTS)
    message = client.chat_completion(messages=messages, stream=False, max_tokens=2000)
    return message.choices[0].message.content


def next_input(_prompt, _completion_a, _completion_b):
    new_dataset = dataset.shuffle()
    row = new_dataset[0]
    messages = row["messages"][:-1]
    completions = [row["response"]]
    completions.append(_get_response(messages))
    completions.append(_get_response(messages))
    random.shuffle(completions)
    return messages, completions.pop(), completions.pop()


if __name__ == "__main__":
    interface = AnnotatorInterFace.for_chat_generation_preference(
        fn_next_input=next_input,
        interactive=[False, True, True],
        dataset_name="dataset-viber-chat-generation-preference-inference-endpoints-battle",
    )
    interface.launch()
