<h1 align="center">
  <a href=""><img src="https://cdn-icons-png.flaticon.com/512/2091/2091395.png" alt="data-viber" width="150"></a>
  <br>
  Data Viber
  <br>
</h1>

<h3 align="center">Avoid the hype, check the vibe!</h2>

I've cooked up Data Viber, a cool set of tools to make your life easier when dealing with data for NLP and image models. Data Viber is all about making your data prep journey smooth and fun. It's **not production-ready** or trying to be all fancy and formal - just a bunch of cool **tools to help you collecting feedback and doing vibe-checks** for data for AI models. Want to see it in action? Just plug it in and start vibing with your data. It's that easy!

Need any tweaks or want to hear more about a specific tool? Just open an issue or give me a shout!

> Great AI engineer: "chickity-check yo' self before you wreck yo' self"

## Installation

I have not published this yet on PyPi, but for now you can install it from the repo.

```bash
pip install git+https://github.com/davidberenstein1957/data-viber.git
```

## How are we vibing?

### Gradio

#### GradioDataCollectorInterface

> Built on top of the `gr.Interface` and `gr.ChatInterface` to collect data and log it to the hub.
> TODO: add a way to collect data from a gr.ChatInterface

Collect data using the `GradioDataCollectorInterface`.

```python
import gradio as gr
from data_viber import GradioDataCollectorInterface

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2

inputs = ["number", gr.Radio(["add", "subtract", "multiply", "divide"]), "number"]
outputs = "number"

interface = GradioDataCollectorInterface(
    fn=calculator,
    inputs=inputs,
    outputs=outputs
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

Collect data from any `gr.Interface`.

```python
interface = gr.Interface(
    fn=calculator,
    inputs=inputs,
    outputs=outputs
)
interface = GradioDataCollectorInterface.from_interface(
   interface=interface,
   dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

Collect data from any `transformers.pipeline`.

```python
from transformers import pipeline
from data_viber import GradioDataCollectorInterface

pipeline = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
interface = GradioDataCollectorInterface.from_pipeline(
    pipeline=pipeline,
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

#### GradioDataAnnotatorInterface

> Built on top of the `gr.GradioDataAnnotatorInterface` to collect and annotate data and log it to the Hub.
> TODO: adding models to the loop (potentially using from_pipeline = interactive)

Annotate data for `text-classification` or `multi-label-text-classification`.

```python
from data_viber import GradioAnnotatorInterFace

texts = ["I hate it!", "I love it!"]
labels = ["positive", "negative"]

interface = GradioAnnotatorInterFace.for_text_classification(
    texts=texts,
    labels=labels,
    dataset_name="<my_hf_org>/<my_dataset>",
    multi_label=False # set to True if you have multi-label data
)
interface.launch()
```

Annotate data for `token-classification`.

```python
from data_viber import GradioAnnotatorInterFace

texts = ["Anthony Bourdain was an amazing chef in New York."]
labels = ["NAME", "LOC"]

interface = GradioAnnotatorInterFace.for_token_classification(
    texts=texts,
    labels=labels,
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

Annotate data for `question-answering`. [WIP]

```python
from data_viber import GradioAnnotatorInterFace

questions = ["Where was Anthony Bourdain located?"]
contexts = ["Anthony Bourdain was an amazing chef in New York."]

interface = GradioAnnotatorInterFace.for_question_answering(
    questions=questions,
    contexts=contexts,
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

Annotate data for `chat`. [WIP]

Annotate data for `chat_preference`. [WIP]

## Contribute and development setup

First, [install PDM](https://pdm-project.org/latest/#installation).

Then, install the environment, this will automatically create a `.venv` virtual env and install the dev environment.

```bash
pdm install
```

Lastly, run pre-commit for formatting on commit.

```bash
pre-commit install
```

Follow this [guide on making first contributions](https://github.com/firstcontributions/first-contributions?tab=readme-ov-file#first-contributions).

## References

### Logo

<a href="https://www.flaticon.com/free-icons/keyboard" title="keyboard icons">Keyboard icons created by srip - Flaticon</a>

### Inspirations

- https://huggingface.co/spaces/davidberenstein1957/llm-human-feedback-collector-chat-interface-dpo
- https://huggingface.co/spaces/davidberenstein1957/llm-human-feedback-collector-chat-interface-kto
- https://medium.com/@oxenai/collecting-data-from-human-feedback-for-generative-ai-ec9e20bf01b9
- https://hamel.dev/notes/llm/finetuning/04_data_cleaning.html
