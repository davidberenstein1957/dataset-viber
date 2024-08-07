<h1 align="center">
  <a href=""><img src="https://cdn-icons-png.flaticon.com/512/2091/2091395.png" alt="data-viber" width="150"></a>
  <br>
  Data Viber
  <br>
</h1>

<h3 align="center">Avoid the hype, check the vibe!</h2>

I've cooked up Data Viber, a cool set of tools to make your life easier when dealing with data for NLP and image models. Data Viber is all about making your data prep journey smooth and fun. It's **not production-ready** or trying to be all fancy and formal - **just a bunch of cool tools to help you in collecting feedback and doing vibe-checks for AI models**. Want to see it in action? Just plug it in and start vibing with your data. It's that easy!

Need any tweaks or want to hear more about a specific tool? Just open an issue or give me a shout!

## Installation

I have not published this yet on PyPi, but for now you can install it from the repo.

```bash
pip install git+https://github.com/davidberenstein1957/data-viber.git
```

## How are we vibing

### GradioDataCollectorInterface

An implementation for any random `transformers.pipeline`.

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

An implementation for any random `gr.Interface`.

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

interface = gr.Interface(
    calculator,
    ["number", gr.Radio(["add", "subtract", "multiply", "divide"]), "number"],
    "number"
)
interface = GradioDataCollectorInterface.from_interface(
    interface=interface,
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

## Contribute and development setup

First, [install PDM](https://pdm-project.org/latest/#installation).

Then, install the environment, this will automatically create a `.venv` virtual env and install the dev environment.

```bash
pdm install
```

Lastly, run pre-commit for formatting on commit.

```
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
