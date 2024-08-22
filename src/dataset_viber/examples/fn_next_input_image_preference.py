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

import concurrent.futures
import io
import os
import random
import time

import requests
from PIL import Image

from dataset_viber import AnnotatorInterFace

HF_TOKEN = os.environ["HF_TOKEN"]
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
DATASET_SERVER_URL = "https://datasets-server.huggingface.co"
DATASET_NAME = "poloclub%2Fdiffusiondb&config=2m_random_1k&split=train"
MODEL_URL = (
    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
)


def retrieve_sample(idx):
    api_url = f"{DATASET_SERVER_URL}/rows?dataset={DATASET_NAME}&offset={idx}&length=1"
    response = requests.get(api_url, headers=HEADERS)
    data = response.json()
    img_url = data["rows"][0]["row"]["image"]["src"]
    prompt = data["rows"][0]["row"]["prompt"]
    return img_url, prompt


def get_rows():
    api_url = f"{DATASET_SERVER_URL}/size?dataset={DATASET_NAME}"
    response = requests.get(api_url, headers=HEADERS)
    num_rows = response.json()["size"]["config"]["num_rows"]
    return num_rows


def generate_response(prompt):
    def _get_response(prompt):
        payload = {
            "inputs": prompt,
        }
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
        if response.status_code != 200:
            time.sleep(5)
            return _get_response(prompt)
        return response

    response = _get_response(prompt)
    image = Image.open(io.BytesIO(response.content))
    return image


def next_input(_prompt, _completion_a, _completion_b):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        random_idx = random.randint(0, get_rows()) - 1
        future = executor.submit(retrieve_sample, random_idx)
        img_url, prompt = future.result()
        generated_image = generate_response(prompt)
    return (prompt, img_url, generated_image)


if __name__ == "__main__":
    interface = AnnotatorInterFace.for_image_generation_preference(
        interactive=False, fn_next_input=next_input
    )
    interface.launch()
