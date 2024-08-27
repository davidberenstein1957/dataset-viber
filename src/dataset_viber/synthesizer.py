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

from typing import Any, Optional

from distilabel.llms import LLM, InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateTextClassificationData, Magpie

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_TOKENIZER_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

DEFAULT_LLM = InferenceEndpointsLLM(
    model_id=DEFAULT_MODEL_ID,
    tokenizer_id=DEFAULT_TOKENIZER_ID,
    magpie_pre_query_template="llama3",
    generation_kwargs={"max_new_tokens": 4000, "temperature": 1},
)


class Synthesizer:
    def __init__(self, next_input: callable):
        self.next_input = next_input

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.next_input(*args, **kwds)

    @classmethod
    def for_text_classification(
        cls,
        task: str,
        llm: Optional[LLM] = None,
        difficulty: Optional[str] = "high school",
        clarity: Optional[str] = "understandable with some effort",
        language: Optional[str] = "english",
    ):
        task_generator = GenerateTextClassificationData(
            llm=llm or DEFAULT_LLM,
            language=language,
            difficulty=difficulty,
            clarity=clarity,
        )
        task_generator.load()

        def next_input(_text, _label):
            inputs: list[dict[str, str]] = [{"task": task}]
            data = next(task_generator.process(inputs))[0]
            return data["input_text"], None

        return cls(next_input)

    @classmethod
    def for_text_generation(cls, task: str, llm: Optional[LLM] = None):
        task_generator = Magpie(
            llm=llm or DEFAULT_LLM,
            n_turns=1,
        )
        task_generator.load()

        task_generator.process([{"system_prompt": task}])

        def next_input(_instruction, _response):
            data = next(task_generator.process([{"system_prompt": task}]))[0]
            return data["instruction"], data["response"]

        return cls(next_input)

    @classmethod
    def for_chat_generation(
        cls,
        task: str,
        llm: Optional[LLM] = None,
        n_turns: int = 2,
    ):
        assert n_turns > 1, "n_turns must be greater than 1"
        task_generator = Magpie(
            llm=llm or DEFAULT_LLM, n_turns=n_turns, end_with_user=True
        )
        task_generator.load()

        def next_input(_conversation, _response):
            data = next(task_generator.process([{"system_prompt": task}]))[0]
            conversation = data["conversation"][:-1]
            response = data["conversation"][-1]["content"]

            return conversation, response

        return cls(next_input)

    @classmethod
    def for_chat_classification(
        cls,
        task: str,
        llm: Optional[LLM] = None,
        n_turns: int = 2,
    ):
        assert n_turns > 1, "n_turns must be greater than 1"
        task_generator = Magpie(
            llm=llm or DEFAULT_LLM, n_turns=n_turns, end_with_user=True
        )
        task_generator.load()

        def next_input(_conversation, _label):
            data = next(task_generator.process([{"system_prompt": task}]))[0]
            return data["conversation"], None

        return cls(next_input)

    @classmethod
    def for_chat_generation_preference(
        cls,
        task: str,
        llm: Optional[LLM] = None,
        n_turns: int = 2,
    ):
        assert n_turns > 1, "n_turns must be greater than 1"
        task_generator = Magpie(
            llm=llm or DEFAULT_LLM, n_turns=n_turns, end_with_user=True
        )
        task_generator.load()

        def next_input(_conversation, _response_1, _response_2):
            data = next(task_generator.process([{"system_prompt": task}]))[0]
            conversation = data["conversation"][:-1]
            response_1 = data["conversation"][-1]["content"]
            response_2 = llm.generate(inputs=[conversation])[-1]["content"]
            return conversation, response_1, response_2

        return cls(next_input)


synthesizer = Synthesizer.for_chat_generation("long IMDB movie reviews")
