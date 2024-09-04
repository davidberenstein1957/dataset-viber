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
import io
import os
import time
import uuid
import warnings
from typing import Any, Optional

import requests
from distilabel.llms import LLM, InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateTextClassificationData, Magpie
from PIL import Image

from dataset_viber._constants import TASK_MAPPING

_DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_GENERATION_KWARGS = {"max_new_tokens": 4000, "temperature": 1, "do_sample": True}
_DEFAULT_LLM = InferenceEndpointsLLM(
    model_id=_DEFAULT_MODEL_ID,
    tokenizer_id=_DEFAULT_MODEL_ID,
    magpie_pre_query_template="llama3",
    generation_kwargs=_GENERATION_KWARGS,
)

_DEFAULT_MODEL_URL = (
    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
)


class _ImageGeneration:
    """
    A class for generating images based on text prompts using a specified model.
    """

    def __init__(self, llm: Optional[str] = None):
        """
        Initialize the _ImageGeneration class.

        Args:
            llm (Optional[str]): The URL of the image generation model. If None, uses the default model.
        """
        self.model_url = llm or _DEFAULT_MODEL_URL
        self.headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

    def process(self, prompt):
        """
        Generate an image based on the given prompt.

        Args:
            prompt (str): The text prompt for image generation.

        Returns:
            PIL.Image.Image: The generated image.
        """

        def _get_response(prompt):
            payload = {"inputs": prompt, "_id": uuid.uuid4().hex}
            response = requests.post(self.model_url, headers=self.headers, json=payload)
            if response.status_code != 200:
                warnings.warn(
                    f"Failed to get response from model. Status code: {response.status_code}. Response: {response.text}"
                )
                time.sleep(5)
                return _get_response(prompt)
            return response

        response = _get_response(prompt)
        image = Image.open(io.BytesIO(response.content))
        return image

    def load(self):
        pass


class Synthesizer:
    """
    A class for synthesizing data for various AI tasks.
    """

    def __init__(self, next_input: callable, prompt_context: str):
        """
        Initialize the Synthesizer class.

        Args:
            next_input (callable): A function to generate the next input.
            prompt_context (str): The context for the prompt.
        """
        self.next_input = next_input
        self.prompt_context = prompt_context

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Call the next_input function with the given arguments.

        Args:
            *args: Positional arguments to pass to next_input.
            **kwds: Keyword arguments to pass to next_input.

        Returns:
            Any: The result of calling next_input.
        """
        return self.next_input(*args, **kwds)

    def batch_synthesize(self, n: int):
        """
        Synthesize a batch of inputs.

        Args:
            n (int): The number of inputs to synthesize.

        Returns:
            list: A list of synthesized inputs.
        """
        batch = [self.next_input(*self.input_columns) for _ in range(n)]
        return list(map(list, zip(*batch)))

    @classmethod
    def _create_synthesizer(
        cls, task_type: str, prompt_context: str, llm: Optional[LLM] = None, **kwargs
    ):
        """
        Create a Synthesizer instance for a specific task type.

        Args:
            task_type (str): The type of task for which to create the synthesizer.
            prompt_context (str): The context for the prompt.
            llm (Optional[LLM]): The language model to use. If None, uses the default model.
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for the specified task.
        """
        if llm:
            warnings.warn(
                "custom LLM passed, make sure to set do_sample=True for generation_kwargs within the llm"
            )

        task_config = TASK_MAPPING[task_type]
        cls.input_columns = task_config["input_columns"] + ["prompt_context"]
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
        """
        Get the appropriate task generator based on the task type.

        Args:
            task_type (str): The type of task.
            llm (LLM): The language model to use.
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Any: An instance of the appropriate task generator.

        Raises:
            ValueError: If an unknown task type is provided.
        """
        if task_type == "text-classification":
            task_generator = GenerateTextClassificationData(llm=llm, **kwargs)
        elif "image" in task_type:
            task_generator = _ImageGeneration()
        else:
            task_generator = Magpie(llm=llm)
            task_generator.set_runtime_parameters(kwargs.get("runtime_parameters", {}))
        task_generator.load()
        return task_generator

    @staticmethod
    def _get_next_input_function(task_type: str, prompt_context: str, task_generator):
        """
        Get the appropriate next_input function based on the task type.

        Args:
            task_type (str): The type of task.
            prompt_context (str): The context for the prompt.
            task_generator: The task generator instance.

        Returns:
            callable: A function that generates the next input for the specified task type.

        Raises:
            ValueError: If an unknown task type is provided.
        """
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
        elif task_type == "image-classification":

            def next_input(_image, _label, _prompt_context):
                _prompt_context = _prompt_context or prompt_context
                image = task_generator.process(_prompt_context)
                return image, None, _prompt_context
        elif task_type == "image-generation":

            def next_input(
                _prompt, _image, _prompt_context
            ):  # -> tuple[Any, Any, Any | str]:# -> tuple[Any, Any, Any | str]:
                _prompt_context = _prompt or _prompt_context or prompt_context
                image = task_generator.process(_prompt_context)
                return _prompt_context, image, _prompt_context

        elif task_type == "image-description":

            def next_input(_image, _description, _prompt_context):
                _prompt_context = _prompt_context or prompt_context
                image = task_generator.process(_prompt_context)
                return image, None, _prompt_context
        elif task_type == "image-generation-preference":

            def next_input(_prompt, _image_1, _image_2, _prompt_context):
                _prompt_context = _prompt or _prompt_context or prompt_context
                image_1 = task_generator.process(_prompt_context)
                image_2 = task_generator.process(_prompt_context)
                return _prompt_context, image_1, image_2, _prompt_context
        elif task_type == "image-question-answering":

            def next_input(_image, _question, _answer, _prompt_context):
                _prompt_context = _prompt_context or prompt_context
                image = task_generator.process(_prompt_context)
                return image, None, None, _prompt_context
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return next_input

    @classmethod
    def for_text_classification(
        cls, prompt_context: str, llm: Optional[LLM] = None, **kwargs
    ) -> "Synthesizer":
        """
        Create a Synthesizer for text classification tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[LLM]): The language model to use. If None, uses the default model.
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for text classification.
        """
        return cls._create_synthesizer(
            "text-classification", prompt_context, llm, **kwargs
        )

    @classmethod
    def for_text_generation(
        cls, prompt_context: str, llm: Optional[LLM] = None
    ) -> "Synthesizer":
        """
        Create a Synthesizer for text generation tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[LLM]): The language model to use. If None, uses the default model.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for text generation.
        """
        return cls._create_synthesizer(
            "text-generation",
            prompt_context,
            llm,
            runtime_parameters={"n_turns": 1, "end_with_user": False},
        )

    @classmethod
    def for_question_answering(
        cls, prompt_context: str, llm: Optional[LLM] = None
    ) -> "Synthesizer":
        raise NotImplementedError

    def for_token_classification(
        cls, prompt_context: str, llm: Optional[LLM] = None
    ) -> "Synthesizer":
        raise NotImplementedError

    @classmethod
    def for_text_generation_preference(
        cls, prompt_context: str, llm: Optional[LLM] = None
    ) -> "Synthesizer":
        """
        Create a Synthesizer for text generation preference tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[LLM]): The language model to use. If None, uses the default model.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for text generation preference.
        """
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
        """
        Create a Synthesizer for chat generation tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[LLM]): The language model to use. If None, uses the default model.
            n_turns (int): The number of turns in the conversation.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for chat generation.
        """
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
        """
        Create a Synthesizer for chat classification tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[LLM]): The language model to use. If None, uses the default model.
            n_turns (int): The number of turns in the conversation.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for chat classification.
        """
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
        """
        Create a Synthesizer for chat generation preference tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[LLM]): The language model to use. If None, uses the default model.
            n_turns (int): The number of turns in the conversation.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for chat generation preference.
        """
        assert n_turns > 1, "n_turns must be greater than 1"
        return cls._create_synthesizer(
            "chat-generation-preference",
            prompt_context,
            llm,
            runtime_parameters={"n_turns": n_turns, "end_with_user": False},
        )

    @classmethod
    def for_image_classification(
        cls, prompt_context: str, llm: Optional[str] = None, **kwargs
    ) -> "Synthesizer":
        """
        Create a Synthesizer for image classification tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[str]): The Hugging Face URL of the image generation model. If None, uses the default model.
                "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for image classification.
        """
        return cls._create_synthesizer(
            "image-classification", prompt_context, llm, **kwargs
        )

    @classmethod
    def for_image_generation(
        cls, prompt_context: str, llm: Optional[str] = None, **kwargs
    ) -> "Synthesizer":
        """
        Create a Synthesizer for image generation tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[str]): The Hugging Face URL of the image generation model. If None, uses the default model.
                "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for image generation.
        """
        return cls._create_synthesizer(
            "image-generation", prompt_context, llm, **kwargs
        )

    @classmethod
    def for_image_description(
        cls, prompt_context: str, llm: Optional[str] = None, **kwargs
    ) -> "Synthesizer":
        """
        Create a Synthesizer for image description tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[str]): The Hugging Face URL of the image generation model. If None, uses the default model.
                "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for image description.
        """
        return cls._create_synthesizer(
            "image-description", prompt_context, llm, **kwargs
        )

    @classmethod
    def for_image_generation_preference(
        cls, prompt_context: str, llm: Optional[str] = None, **kwargs
    ) -> "Synthesizer":
        """
        Create a Synthesizer for image generation preference tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[str]): The Hugging Face URL of the image generation model. If None, uses the default model.
                "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for image generation preference.
        """
        return cls._create_synthesizer(
            "image-generation-preference", prompt_context, llm, **kwargs
        )

    @classmethod
    def for_image_question_answering(
        cls, prompt_context: str, llm: Optional[str] = None, **kwargs
    ) -> "Synthesizer":
        """
        Create a Synthesizer for image question answering tasks.

        Args:
            prompt_context (str): The context for the prompt.
            llm (Optional[str]): The Hugging Face URL of the image generation model. If None, uses the default model.
                "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            **kwargs: Additional keyword arguments for task configuration.

        Returns:
            Synthesizer: An instance of the Synthesizer class configured for image question answering.
        """
        return cls._create_synthesizer(
            "image-question-answering", prompt_context, llm, **kwargs
        )
