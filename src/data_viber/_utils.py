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

import inspect


def _get_init_arg_names(cls) -> list[str]:
    init_signature = inspect.signature(cls.__init__)
    return [param.name for param in init_signature.parameters.values()]


def _get_init_payload(cls) -> dict:
    payload = cls.__dict__
    payload["inputs"] = payload["input_components"]
    payload["outputs"] = payload["output_components"]
    return {
        key: value for key, value in payload.items() if key in _get_init_arg_names(cls)
    }
