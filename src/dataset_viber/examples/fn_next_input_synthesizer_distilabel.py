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
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_text_generation(
    task_description="An expert in the field of AI"
)

interface = AnnotatorInterFace.for_text_generation(fn_next_input=synthesizer)
interface.launch()
