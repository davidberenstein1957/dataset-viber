<h1 align="center">
  <a href=""><img src="https://cdn-icons-png.flaticon.com/512/2091/2091395.png" alt="data-viber" width="150"></a>
  <br>
  Data Viber
  <br>
</h1>

<h3 align="center">Avoid the hype, check the vibe!</h2>

I've cooked up Data Viber, a cool set of tools to make your life easier when dealing with data for NLP and image models. Data Viber is all about making your data prep journey smooth and fun. It's **not production-ready** or trying to be all fancy and formal - just a bunch of cool **tools to help you collect feedback and do vibe-checks** for data for AI models. Want to see it in action? Just plug it in and start vibing with your data. It's that easy! Vibing

- **CollectorInterface**: Lazily collect data without human annotation.
- **AnnotatorInterface**: Work through your data and annotate it with ease.
- **ExplorerInterface**: Explore your data distribution and similarity.

Need any tweaks or want to hear more about a specific tool? Just open an issue or give me a shout!

> [!TIP]
> All tools also run in `.ipynb` notebooks!

## Installation

I have not published this yet on PyPi, but for now, you can install it from the repo.

```bash
pip install git+https://github.com/davidberenstein1957/data-viber.git
```

## How are we vibing?

### CollectorInterface

> Built on top of the `gr.Interface` and `gr.ChatInterface` to lazily collect data for interactions automatically.
> TODO: add a way to collect data from a gr.ChatInterface

https://github.com/user-attachments/assets/4ddac8a1-62ab-4b3b-9254-f924f5898075

<details>
<summary><code>CollectorInterface</code></summary>

```python
import gradio as gr
from data_viber import CollectorInterface

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

interface = CollectorInterface(
    fn=calculator,
    inputs=inputs,
    outputs=outputs,
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

</details>

<details>
<summary><code>CollectorInterface.from_interface</code></summary>

```python
interface = gr.Interface(
    fn=calculator,
    inputs=inputs,
    outputs=outputs
)
interface = CollectorInterface.from_interface(
   interface=interface,
   dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

</details>

<details>
<summary><code>CollectorInterface.from_pipeline</code></summary>

```python
from transformers import pipeline
from data_viber import CollectorInterface

pipeline = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
interface = CollectorInterface.from_pipeline(
    pipeline=pipeline,
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

</details>

### AnnotatorInterface

> Built on top of the `CollectorInterface` to collect and annotate data and log it to the Hub.

> TODO: annotate - models to the loop (potentially using from_pipeline = interactive)
> TODO: annotate - counters for the number of annotations
> TODO: data - a state based on csv or remote dataset
> TODO: data - local datasets saver / loader from csv
> TODO: data - a way to show input-data and output-data in the interface

https://github.com/user-attachments/assets/57d89edf-ae40-4942-a20a-bf8443100b66

#### Text

<details>
<summary><code>text-classification</code> and <code>multi-label-text-classification</code></summary>

```python
from data_viber import AnnotatorInterFace

texts = [
    "Anthony Bourdain was an amazing chef!",
    "Anthony Bourdain was a terrible tv persona!"
]
labels = ["positive", "negative"]

interface = AnnotatorInterFace.for_text_classification(
    texts=texts,
    labels=labels,
    dataset_name=None, # "<my_hf_org>/<my_dataset>" if you want to log to the hub
    multi_label=False # True if you have multi-label data
)
interface.launch()
```

</details>

<details>
<summary><code>token-classification</code></summary>

```python
from data_viber import AnnotatorInterFace

texts = ["Anthony Bourdain was an amazing chef in New York."]
labels = ["NAME", "LOC"]

interface = AnnotatorInterFace.for_token_classification(
    texts=texts,
    labels=labels,
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>extractive-question-answering</code></summary>

```python
from data_viber import AnnotatorInterFace

questions = ["Where was Anthony Bourdain located?"]
contexts = ["Anthony Bourdain was an amazing chef in New York."]

interface = AnnotatorInterFace.for_question_answering(
    questions=questions,
    contexts=contexts,
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>text-generation</code> or <code>translation</code></summary>

```python
from data_viber import AnnotatorInterFace

source = ["Tell me something about Anthony Bourdain."]
target = ["Anthony Michael Bourdain was an American celebrity chef, author, and travel documentarian."]

interface = AnnotatorInterFace.for_text_generation(
    source=source,
    target=target, # optional to show initial target
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)

```
</details>

#### Chat

Annotate data for `chat`. [WIP]

Annotate data for `chat_preference`. [WIP]

#### Image and multi-modal

I recommend uploading the files files to a cloud storage and using the remote URL to avoid any issues. This can be done [using Hugging Face Datasets](https://huggingface.co/docs/datasets/en/image_load#local-files).

<details>
<summary><code>image-classification</code> and <code>multi-label-text-classification</code></summary>

```python
from data_viber import AnnotatorInterFace

images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg"
]
labels = ["anthony-bourdain", "not-anthony-bourdain"]

interface = AnnotatorInterFace.for_image_classification(
    images=images,
    labels=labels,
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
```

</details>

<details>
<summary><code>image-2-text</code> or <code>image-description</code></summary>

```python
from data_viber import AnnotatorInterFace

images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg"
]
description = ["Anthony Bourdain laughing", "David Chang wearing a suit"]

interface = AnnotatorInterFace.for_image_description(
    images=images,
    descriptions=descriptions, # optional to show initial descriptions
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>image-question-answering</code> or <code>visual-question-asnwering</code></summary>

```python
from data_viber import AnnotatorInterFace

images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg"
]
questions = ["Who is this?", "What is he wearing?"]
answers = ["Anthony Bourdain", "a suit"]

interface = AnnotatorInterFace.for_image_question_answering(
    images=images,
    questions=questions, # optional to show initial questions
    answers=answers, # optional to show initial answers
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

### ExplorerInterface

> Built on top of the `gr.ScatterPlot`, `gr.DataFrame`, `umap-learn`, and `sentence-transformers` to understand the data distribution and similarity.
> TODO: create basic explorer for text data
> TODO: add score representation
> TODO: add filters for categories / scores
> TODO: add image support
> TODO: create label explorer

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
