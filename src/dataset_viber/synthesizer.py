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

import warnings
from typing import Any, Optional

from distilabel.llms import LLM, InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateTextClassificationData, Magpie

from dataset_viber._constants import TASK_MAPPING

_DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_GENERATION_KWARGS = {"max_new_tokens": 4000, "temperature": 1, "do_sample": True}
_DEFAULT_LLM = InferenceEndpointsLLM(
    model_id=_DEFAULT_MODEL_ID,
    tokenizer_id=_DEFAULT_MODEL_ID,
    magpie_pre_query_template="llama3",
    generation_kwargs=_GENERATION_KWARGS,
)


class Synthesizer:
    def __init__(self, next_input: callable, prompt_context: str):
        self.next_input = next_input
        self.prompt_context = prompt_context

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.next_input(*args, **kwds)

    def batch_synthesize(self, n: int):
        batch = [self.next_input(*self.input_columns) for _ in range(n)]
        return list(map(list, zip(*batch)))

    @classmethod
    def _create_synthesizer(
        cls, task_type: str, prompt_context: str, llm: Optional[LLM] = None, **kwargs
    ):
        if llm:
            warnings.warn(
                "custom LLM passed, make sure to set do_sample=True for generation_kwargs within the llm"
            )

        task_config = TASK_MAPPING[task_type]
        cls.input_columns = task_config["input_columns"]
        cls.output_columns = task_config["output_columns"]

        task_generator = cls._get_task_generator(
            task_type, llm or _DEFAULT_LLM, **kwargs
        )
        next_input = cls._get_next_input_function(
            task_type, prompt_context, task_generator
        )

        return cls(next_input, prompt_context)

    @staticmethod
    def _get_task_generator(task_type: str, llm: LLM, **kwargs):
        if task_type == "text-classification":
            task_generator = GenerateTextClassificationData(llm=llm, **kwargs)
        else:
            task_generator = Magpie(llm=llm)
            task_generator.set_runtime_parameters(kwargs.get("runtime_parameters", {}))
        task_generator.load()
        return task_generator

    @staticmethod
    def _get_next_input_function(task_type: str, prompt_context: str, task_generator):
        if task_type == "text-classification":

            def next_input(_text, _label, _prompt_context):
                _prompt_context = _prompt_context or prompt_context
                inputs = [{"task": _prompt_context}]
                data = next(task_generator.process(inputs))[0]
                return data["input_text"], None, _prompt_context
        elif task_type in ["text-generation", "chat-generation"]:

            def next_input(_instruction, _response, _prompt_context):
                _prompt_context = _prompt_context or prompt_context
                data = next(
                    task_generator.process([{"system_prompt": _prompt_context}])
                )[0]
                if task_type == "text-generation":
                    return data["instruction"], data["response"], _prompt_context
                else:
                    conversation = data["conversation"][:-1]
                    response = data["conversation"][-1]["content"]
                    return conversation, response, _prompt_context
        elif task_type in ["text-generation-preference", "chat-generation-preference"]:

            def next_input(_conversation, _response_1, _response_2, _prompt_context):
                _prompt_context = _prompt_context or prompt_context
                data = next(
                    task_generator.process([{"system_prompt": _prompt_context}])
                )[0]
                if task_type == "text-generation-preference":
                    response_2 = task_generator.llm.generate(
                        inputs=[[{"role": "user", "content": data["instruction"]}]],
                        **_GENERATION_KWARGS,
                    )[0][0]
                    return (
                        data["instruction"],
                        data["response"],
                        response_2,
                        _prompt_context,
                    )
                else:
                    conversation = data["conversation"][:-1]
                    response_1 = data["conversation"][-1]["content"]
                    response_2 = task_generator.llm.generate(
                        inputs=[conversation], **_GENERATION_KWARGS
                    )[0][0]
                    return conversation, response_1, response_2, _prompt_context
        elif task_type == "chat-classification":

            def next_input(_conversation, _label, _prompt_context):
                _prompt_context = _prompt_context or prompt_context
                data = next(
                    task_generator.process([{"system_prompt": _prompt_context}])
                )[0]
                return data["conversation"], None, _prompt_context
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return next_input

    @classmethod
    def for_text_classification(
        cls, prompt_context: str, llm: Optional[LLM] = None, **kwargs
    ) -> "Synthesizer":
        return cls._create_synthesizer(
            "text-classification", prompt_context, llm, **kwargs
        )

    @classmethod
    def for_text_generation(
        cls, prompt_context: str, llm: Optional[LLM] = None
    ) -> "Synthesizer":
        return cls._create_synthesizer(
            "text-generation",
            prompt_context,
            llm,
            runtime_parameters={"n_turns": 1, "end_with_user": False},
        )

    @classmethod
    def for_text_generation_preference(
        cls, prompt_context: str, llm: Optional[LLM] = None
    ) -> "Synthesizer":
        return cls._create_synthesizer(
            "text-generation-preference",
            prompt_context,
            llm,
            runtime_parameters={"n_turns": 1, "end_with_user": False},
        )

    @classmethod
    def for_chat_generation(
        cls, prompt_context: str, llm: Optional[LLM] = None, n_turns: int = 2
    ) -> "Synthesizer":
        assert n_turns > 1, "n_turns must be greater than 1"
        return cls._create_synthesizer(
            "chat-generation",
            prompt_context,
            llm,
            runtime_parameters={"n_turns": n_turns, "end_with_user": False},
        )

    @classmethod
    def for_chat_classification(
        cls, prompt_context: str, llm: Optional[LLM] = None, n_turns: int = 2
    ) -> "Synthesizer":
        assert n_turns > 1, "n_turns must be greater than 1"
        return cls._create_synthesizer(
            "chat-classification",
            prompt_context,
            llm,
            runtime_parameters={"n_turns": n_turns, "end_with_user": False},
        )

    @classmethod
    def for_chat_generation_preference(
        cls, prompt_context: str, llm: Optional[LLM] = None, n_turns: int = 2
    ) -> "Synthesizer":
        assert n_turns > 1, "n_turns must be greater than 1"
        return cls._create_synthesizer(
            "chat-generation-preference",
            prompt_context,
            llm,
            runtime_parameters={"n_turns": n_turns, "end_with_user": False},
        )
