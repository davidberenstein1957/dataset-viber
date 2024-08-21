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

from dataset_viber import AnnotatorInterFace

texts = [
    "Anthony Bourdain was an amazing chef!",
    "Anthony Bourdain was a terrible tv persona!",
]
labels = ["positive", "negative"]

interface = AnnotatorInterFace.for_text_classification(
    texts=texts,
    labels=labels,
    csv_logger=True,  # True if you want to log to a CSV file
)
interface.launch()
