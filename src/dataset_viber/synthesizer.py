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
from typing import Any, Dict, Optional

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
        """Initialize the Synthesizer with a callable for input processing."""
        self.next_input = next_input
        self.prompt_context = prompt_context

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Invoke the next input callable with provided arguments."""
        return self.next_input(*args, **kwds)

    def batch_synthesize(self, n: int):
        """Batch the input callable with provided arguments."""
        batch = []
        for _ in range(n):
            batch.append(self.next_input(*self.input_columns))
        return [list(x) for x in zip(*batch)]

    @classmethod
    def for_text_classification(
        cls,
        prompt_context: str,
        llm: Optional[LLM] = None,
        difficulty: Optional[str] = "high school",
        clarity: Optional[str] = "understandable with some effort",
        language: Optional[str] = "english",
    ) -> "Synthesizer":
        """Create a Synthesizer for text classification tasks.

        Args:
            prompt_context: The description of the task
            llm: The distilabel LLM to use for the task.
            difficulty: The difficulty of the task.
            clarity: The clarity of the task.
            language: The language of the task.

        Examples:

            ```python
            synthesizer = Synthesizer.for_text_classification(
                prompt_context="A phone company customer support expert"
                llm=distilabel.llms.vLLM(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    generation_kwargs={"max_new_tokens": 4000, "temperature": 1, "do_sample": True},
                )
            )
            ```

        Returns:
            A Synthesizer for text classification tasks.
        """
        if llm:
            warnings.warn(
                "custom LLM passed, make sure to set do_sample=True for generation_kwargs within the llm"
            )
        task_config = TASK_MAPPING["text-classification"]
        cls.input_columns = task_config["input_columns"]
        cls.output_columns = task_config["output_columns"]

        task_generator = GenerateTextClassificationData(
            llm=llm or _DEFAULT_LLM,
            language=language,
            difficulty=difficulty,
            clarity=clarity,
        )
        task_generator.load()

        def next_input(_text, _label, _prompt_context):
            _prompt_context = _prompt_context or prompt_context
            inputs: list[dict[str, str]] = [{"task": _prompt_context}]
            data = next(task_generator.process(inputs))[0]
            return data["input_text"], None, _prompt_context

        return cls(next_input, prompt_context)

    @classmethod
    def for_text_generation(
        cls, prompt_context: str, llm: Optional[LLM] = None
    ) -> "Synthesizer":
        """Create a Synthesizer for text generation tasks.

        Args:
            prompt_context: The description of the task.
            llm: The distilabel LLM to use for the task.
                Note that the LLM must support the Magpie chat and requires a tokenizer.

        Examples:

            ```python
            synthesizer = Synthesizer.for_text_generation(
                prompt_context="A phone company customer support expert"
                llm=distilabel.llms.vLLM(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    generation_kwargs={"max_new_tokens": 4000, "temperature": 1, "do_sample": True},
                    magpie_pre_query_template="llama3",
                )
            )
            ```

        Returns:
            A Synthesizer for text generation tasks.
        """
        if llm:
            warnings.warn(
                "custom LLM passed, make sure to set do_sample=True for generation_kwargs within the llm"
            )
        task_generator = Magpie(llm=llm or _DEFAULT_LLM)
        task_generator.load()
        task_generator.set_runtime_parameters({"n_turns": 1, "end_with_user": False})

        def next_input(_instruction, _response, _prompt_context):
            _prompt_context = _prompt_context or prompt_context
            data = next(task_generator.process([{"system_prompt": _prompt_context}]))[0]
            return data["instruction"], data["response"], _prompt_context

        return cls(next_input, prompt_context)

    @classmethod
    def for_text_generation_preference(
        cls,
        prompt_context: str,
        llm: Optional[LLM] = None,
    ) -> "Synthesizer":
        """Create a Synthesizer for chat generation with preference tasks.

        Args:
            prompt_context: The description of the task.
            llm: The distilabel LLM to use for the task.
                Note that the LLM must support the Magpie chat and requires a tokenizer.

        Examples:

            ```python
            synthesizer = Synthesizer.for_chat_generation_preference(
                prompt_context="A phone company customer support expert"
                llm=distilabel.llms.vLLM(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    generation_kwargs={"max_new_tokens": 4000, "temperature": 1, "do_sample": True},
                    magpie_pre_query_template="llama3",
                )
        Returns:
            A Synthesizer for chat generation with preference tasks.
        """
        if llm:
            warnings.warn(
                "custom LLM passed, make sure to set do_sample=True for generation_kwargs within the llm"
            )
        task_generator = Magpie(llm=llm or _DEFAULT_LLM)
        task_generator.load()
        task_generator.set_runtime_parameters({"n_turns": 1, "end_with_user": False})

        def next_input(_conversation, _response_1, _response_2, _prompt_context):
            _prompt_context = _prompt_context or prompt_context
            data: Dict[str, Any] = next(
                task_generator.process([{"system_prompt": _prompt_context}])
            )[0]
            response_2 = task_generator.llm.generate(
                inputs=[
                    [
                        {
                            "role": "user",
                            "content": data["instruction"],
                        }
                    ]
                ],
                **_GENERATION_KWARGS,
            )[0][0]
            return data["instruction"], data["response"], response_2, _prompt_context

        return cls(next_input, prompt_context)

    @classmethod
    def for_chat_generation(
        cls,
        prompt_context: str,
        llm: Optional[LLM] = None,
        n_turns: int = 2,
    ) -> "Synthesizer":
        """Create a Synthesizer for chat generation tasks.

        Args:
            prompt_context: The description of the task.
            llm: The distilabel LLM to use for the task.
                Note that the LLM must support the Magpie chat and requires a tokenizer.
            n_turns: The number of turns in the chat.

        Examples:

            ```python
            synthesizer = Synthesizer.for_chat_generation(
                prompt_context="A phone company customer support expert"
                llm=distilabel.llms.vLLM(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    generation_kwargs={"max_new_tokens": 4000, "temperature": 1, "do_sample": True},
                    magpie_pre_query_template="llama3",
                )
            )
            ```

        Returns:
            A Synthesizer for chat generation tasks.
        """
        if llm:
            warnings.warn(
                "custom LLM passed, make sure to set do_sample=True for generation_kwargs within the llm"
            )
        task_config = TASK_MAPPING["chat-generation"]
        cls.input_columns = task_config["input_columns"]
        cls.output_columns = task_config["output_columns"]
        
        assert n_turns > 1, "n_turns must be greater than 1"
        task_generator = Magpie(llm=llm or _DEFAULT_LLM)
        task_generator.load()
        task_generator.set_runtime_parameters(
            {"n_turns": n_turns, "end_with_user": False}
        )

        def next_input(_conversation, _response, _prompt_context):
            _prompt_context = _prompt_context or prompt_context
            data = next(task_generator.process([{"system_prompt": _prompt_context}]))[0]
            conversation = data["conversation"][:-1]
            response = data["conversation"][-1]["content"]

            return conversation, response, _prompt_context

        return cls(next_input, prompt_context)

    @classmethod
    def for_chat_classification(
        cls,
        prompt_context: str,
        llm: Optional[LLM] = None,
        n_turns: int = 2,
    ) -> "Synthesizer":
        """Create a Synthesizer for chat classification tasks.

        Args:
            prompt_context: The description of the task.
            llm: The distilabel LLM to use for the task.
                Note that the LLM must support the Magpie chat and requires a tokenizer.
            n_turns: The number of turns in the chat.

        Examples:

            ```python
            synthesizer = Synthesizer.for_chat_classification(
                prompt_context="A phone company customer support expert"
                llm=distilabel.llms.vLLM(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    generation_kwargs={"max_new_tokens": 4000, "temperature": 1, "do_sample": True},
                    magpie_pre_query_template="llama3",
                )

        Returns:
            A Synthesizer for chat classification tasks.
        """
        task_config = TASK_MAPPING["chat-classification"]
        cls.input_columns = task_config["input_columns"]
        cls.output_columns = task_config["output_columns"]
      
        assert n_turns > 1, "n_turns must be greater than 1"
        task_generator = Magpie(llm=llm or _DEFAULT_LLM)
        task_generator.load()
        task_generator.set_runtime_parameters(
            {"n_turns": n_turns, "end_with_user": False}
        )

        def next_input(_conversation, _label, _prompt_context):
            _prompt_context = _prompt_context or prompt_context
            data = next(task_generator.process([{"system_prompt": _prompt_context}]))[0]
            return data["conversation"], None, _prompt_context

        return cls(next_input, prompt_context)

    @classmethod
    def for_chat_generation_preference(
        cls,
        prompt_context: str,
        llm: Optional[LLM] = None,
        n_turns: int = 2,
    ) -> "Synthesizer":
        """Create a Synthesizer for chat generation with preference tasks.

        Args:
            prompt_context: The description of the task.
            llm: The distilabel LLM to use for the task.
                Note that the LLM must support the Magpie chat and requires a tokenizer.
            n_turns: The number of turns in the chat.

        Examples:

            ```python
            synthesizer = Synthesizer.for_chat_generation_preference(
                prompt_context="A phone company customer support expert"
                llm=distilabel.llms.vLLM(
                    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    generation_kwargs={"max_new_tokens": 4000, "temperature": 1, "do_sample": True},
                    magpie_pre_query_template="llama3",
                )
        Returns:
            A Synthesizer for chat generation with preference tasks.
        """
        if llm:
            warnings.warn(
                "custom LLM passed, make sure to set do_sample=True for generation_kwargs within the llm"
            )
        task_config = TASK_MAPPING["chat-generation-preference"]
        cls.input_columns = task_config["input_columns"]
        cls.output_columns = task_config["output_columns"]

        assert n_turns > 1, "n_turns must be greater than 1"
        task_generator = Magpie(llm=llm or _DEFAULT_LLM)
        task_generator.load()
        task_generator.set_runtime_parameters(
            {"n_turns": n_turns, "end_with_user": False}
        )

        def next_input(_conversation, _response_1, _response_2, _prompt_context):
            _prompt_context = _prompt_context or prompt_context
            data = next(task_generator.process([{"system_prompt": _prompt_context}]))[0]
            conversation = data["conversation"][:-1]
            response_1 = data["conversation"][-1]["content"]
            response_2 = task_generator.llm.generate(
                inputs=[conversation], **_GENERATION_KWARGS
            )[0][0]
            return conversation, response_1, response_2, _prompt_context

        return cls(next_input, prompt_context)
