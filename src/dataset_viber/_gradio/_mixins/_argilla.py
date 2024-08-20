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

import argilla as rg


class ArgillaMixin:
    def get_argilla_dataset(self):
        class MockClient:
            a = "dataset"

        client = MockClient()
        client.api.datasets = "mock"
        return rg.Dataset(
            name="fake-dataset",
            settings=self._get_argilla_settings(),
        )

    def _get_argilla_settings(self):
        if self.task == "text-classification":
            return rg.Settings(
                fields=[rg.TextField(name="text")],
                questions=[rg.LabelQuestion(name="label", labels=self.labels)],
            )
        elif self.task == "text-classification-multi-label":
            return rg.Settings(
                fields=[rg.TextField(name="text")],
                questions=[rg.MultiLabelQuestion(name="label", labels=self.labels)],
            )
        elif self.task == "token-classification":
            raise NotImplementedError
        elif self.task == "question-answering":
            raise NotImplementedError
        elif self.task == "text-generation":
            return rg.Settings(
                fields=[rg.TextField(name="prompt")],
                questions=[rg.TextQuestion(name="completion")],
            )
        elif self.task == "text-generation-preference":
            return rg.Settings(
                fields=[
                    rg.TextField(name="prompt"),
                    rg.TextField(name="chosen"),
                    rg.TextField(name="rejected"),
                ],
                questions=[
                    rg.LabelQuestion(name="flag", labels=["A", "B", "tie"]),
                    rg.TextField(name="reason", required=False),
                ],
            )
        elif self.task == "chat-classification":
            raise NotImplementedError
        elif self.task == "chat-classification-multi-label":
            raise NotImplementedError
        elif self.task == "chat-generation":
            raise NotImplementedError
        elif self.task == "chat-generation-preference":
            raise NotImplementedError
        elif self.task == "image-classification":
            raise NotImplementedError
        elif self.task == "image-classification-multi-label":
            raise NotImplementedError
        elif self.task == "image-generation":
            raise NotImplementedError
        elif self.task == "image-description":
            raise NotImplementedError
        elif self.task == "image-generation-preference":
            raise NotImplementedError
        elif self.task == "image-question-answering":
            raise NotImplementedError
        else:
            raise NotImplementedError
