<h1 align="center">
  <a href=""><img src="https://cdn-icons-png.flaticon.com/512/2091/2091395.png" alt="dataset-viber" width="150"></a>
  <br>
  Dataset Viber
  <br>
</h1>

<h3 align="center">Avoid the hype, check the vibe!</h2>

I've cooked up Dataset Viber, a cool set of tools to make your life easier when dealing with data for AI models. Dataset Viber is all about making your data prep journey smooth and fun. It's **not for team collaboration or production**, nor trying to be all fancy and formal - just a bunch of **cool tools to help you collect feedback and do vibe-checks** as an AI engineer or lover. Want to see it in action? Just plug it in and start vibing with your data. It's that easy!

- **CollectorInterface**: Lazily collect data of model interactions without human annotation.
- **AnnotatorInterface**: Walk through your data and annotate it with models in the loop.
- **Synthesizer**: Synthesize data with `distilabel` in the loop.
- **BulkInterface**: Explore your data distribution and annotate in bulk.

Need any tweaks or want to hear more about a specific tool? Just [open an issue](https://github.com/davidberenstein1957/dataset-viber/issues/new) or give me a shout!

> [!NOTE]
>
> - Data is logged to a local CSV or directly to the Hugging Face Hub.
> - All tools also run in `.ipynb` notebooks.
> - Models in the loop through `fn_model`.
> - Input with custom data streamers or pre-built `Synthesizer` classes with the `fn_next_input` argument.
> - It supports various tasks for `text`, `chat` and `image` modalities.
> - Import and export from the Hugging Face Hub or CSV files.

> [!TIP]
>
> - Code examples: [src/dataset_viber/examples](https://github.com/davidberenstein1957/dataset-viber/tree/main/src/dataset_viber/examples).
> - Hub examples: [https://huggingface.co/dataset-viber](https://huggingface.co/dataset-viber).

## Installation

You can install the package via pip:

```bash
pip install dataset-viber
```

Or install `Synthesizer` dependencies. Note, that the `Synthesizer` relies on `distilabel[hf-inference-endpoints]`, but you can use other [LLMs available to distilabel](https://distilabel.argilla.io) too, like for example `distilabel[ollama]`.

```bash
pip install dataset-viber[synthesizer]
```

Or install `BulkInterface` dependencies:

```bash
pip install dataset-viber[bulk]
```

## How are we vibing?

### CollectorInterface

> Built on top of the `gr.Interface` and `gr.ChatInterface` to lazily collect data for interactions automatically.

<https://github.com/user-attachments/assets/4ddac8a1-62ab-4b3b-9254-f924f5898075>

[Hub dataset](https://huggingface.co/datasets/davidberenstein1957/dataset-viber-token-classification)

<details>
<summary><code>CollectorInterface</code></summary>

```python
import gradio as gr
from dataset_viber import CollectorInterface

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
    csv_logger=False, # True if you want to log to a CSV
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
   csv_logger=False, # True if you want to log to a CSV
   dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

</details>

<details>
<summary><code>CollectorInterface.from_pipeline</code></summary>

```python
from transformers import pipeline
from dataset_viber import CollectorInterface

pipeline = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
interface = CollectorInterface.from_pipeline(
    pipeline=pipeline,
    csv_logger=False, # True if you want to log to a CSV
    dataset_name="<my_hf_org>/<my_dataset>"
)
interface.launch()
```

</details>

### AnnotatorInterface

> Built on top of the `CollectorInterface` to collect and annotate data and log it to the Hub.


#### Text

https://github.com/user-attachments/assets/d1abda66-9972-4c60-89d2-7626f5654f15

[Hub dataset](https://huggingface.co/datasets/davidberenstein1957/dataset-viber-text-classification)

<details>
<summary><code>text-classification</code>/<code>multi-label-text-classification</code></summary>

```python
from dataset_viber import AnnotatorInterFace

texts = [
    "Anthony Bourdain was an amazing chef!",
    "Anthony Bourdain was a terrible tv persona!"
]
labels = ["positive", "negative"]

interface = AnnotatorInterFace.for_text_classification(
    texts=texts,
    labels=labels,
    multi_label=False, # True if you have multi-label data
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>token-classification</code></summary>

```python
from dataset_viber import AnnotatorInterFace

texts = ["Anthony Bourdain was an amazing chef in New York."]
labels = ["NAME", "LOC"]

interface = AnnotatorInterFace.for_token_classification(
    texts=texts,
    labels=labels,
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>extractive-question-answering</code></summary>

```python
from dataset_viber import AnnotatorInterFace

questions = ["Where was Anthony Bourdain located?"]
contexts = ["Anthony Bourdain was an amazing chef in New York."]

interface = AnnotatorInterFace.for_question_answering(
    questions=questions,
    contexts=contexts,
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>text-generation</code>/<code>translation</code>/<code>completion</code></summary>

```python
from dataset_viber import AnnotatorInterFace

prompts = ["Tell me something about Anthony Bourdain."]
completions = ["Anthony Michael Bourdain was an American celebrity chef, author, and travel documentarian."]

interface = AnnotatorInterFace.for_text_generation(
    prompts=prompts, # source
    completions=completions, # optional to show initial completion / target
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>text-generation-preference</code></summary>

```python
from dataset_viber import AnnotatorInterFace

prompts = ["Tell me something about Anthony Bourdain."]
completions_a = ["Anthony Michael Bourdain was an American celebrity chef, author, and travel documentarian."]
completions_b = ["Anthony Michael Bourdain was an cool guy that knew how to cook."]

interface = AnnotatorInterFace.for_text_generation_preference(
    prompts=prompts,
    completions_a=completions_a,
    completions_b=completions_b,
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

#### Chat and multi-modal chat

https://github.com/user-attachments/assets/fe7f0139-95a3-40e8-bc03-e37667d4f7a9

[Hub dataset](https://huggingface.co/datasets/davidberenstein1957/dataset-viber-chat-generation-preference)

> [!TIP]
> I recommend uploading the files files to a cloud storage and using the remote URL to avoid any issues. This can be done [using Hugging Face Datasets](https://huggingface.co/docs/datasets/en/image_load#local-files). As shown in [utils](#utils). Additionally [GradioChatbot](https://www.gradio.app/docs/gradio/chatbot#behavior) shows how to use the chatbot interface for multi-modal.

<details>
<summary><code>chat-classification</code></summary>

```python
from dataset_viber import AnnotatorInterFace

prompts = [
    [
        {
            "role": "user",
            "content": "Tell me something about Anthony Bourdain."
        },
        {
            "role": "assistant",
            "content": "Anthony Michael Bourdain was an American celebrity chef, author, and travel documentarian."
        }
    ]
]

interface = AnnotatorInterFace.for_chat_classification(
    prompts=prompts,
    labels=["toxic", "non-toxic"],
    multi_label=False, # True if you have multi-label data
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>chat-generation</code></summary>

```python
from dataset_viber import AnnotatorInterFace

prompts = [
    [
        {
            "role": "user",
            "content": "Tell me something about Anthony Bourdain."
        }
    ]
]

completions = [
    "Anthony Michael Bourdain was an American celebrity chef, author, and travel documentarian.",
]

interface = AnnotatorInterFace.for_chat_generation(
    prompts=prompts,
    completions=completions,
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>chat-generation-preference</code></summary>

```python
from dataset_viber import AnnotatorInterFace

prompts = [
    [
        {
            "role": "user",
            "content": "Tell me something about Anthony Bourdain."
        }
    ]
]
completions_a = [
    "Anthony Michael Bourdain was an American celebrity chef, author, and travel documentarian.",
]
completions_b = [
    "Anthony Michael Bourdain was an cool guy that knew how to cook."
]

interface = AnnotatorInterFace.for_chat_generation_preference(
    prompts=prompts,
    completions_a=completions_a,
    completions_b=completions_b,
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

#### Image and multi-modal

<https://github.com/user-attachments/assets/57d89edf-ae40-4942-a20a-bf8443100b66>

[Hub dataset](https://huggingface.co/datasets/davidberenstein1957/dataset-viber-image-question-answering)

> [!TIP]
> I recommend uploading the files files to a cloud storage and using the remote URL to avoid any issues. This can be done [using Hugging Face Datasets](https://huggingface.co/docs/datasets/en/image_load#local-files). As shown in [utils](#utils).

<details>
<summary><code>image-classification</code>/<code>multi-label-image-classification</code></summary>

```python
from dataset_viber import AnnotatorInterFace

images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg"
]
labels = ["anthony-bourdain", "not-anthony-bourdain"]

interface = AnnotatorInterFace.for_image_classification(
    images=images,
    labels=labels,
    multi_label=False, # True if you have multi-label data
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>image-generation</code></summary>

```python
from dataset_viber import AnnotatorInterFace

prompts = [
    "Anthony Bourdain laughing",
    "David Chang wearing a suit"
]
images = [
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
]

interface = AnnotatorInterFace.for_image_generation(
    prompts=prompts,
    completions=images,
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)

interface.launch()
```

</details>

<details>
<summary><code>image-description</code></summary>

```python
from dataset_viber import AnnotatorInterFace

images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg"
]
descriptions = ["Anthony Bourdain laughing", "David Chang wearing a suit"]

interface = AnnotatorInterFace.for_image_description(
    images=images,
    descriptions=descriptions, # optional to show initial descriptions
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>image-question-answering</code>/<code>visual-question-answering</code></summary>

```python
from dataset_viber import AnnotatorInterFace

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
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

<details>
<summary><code>image-generation-preference</code></summary>

```python
from dataset_viber import AnnotatorInterFace

prompts = [
    "Anthony Bourdain laughing",
    "David Chang wearing a suit"
]

images_a = [
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
]

images_b = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Anthony_Bourdain_Peabody_2014b.jpg/440px-Anthony_Bourdain_Peabody_2014b.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/85/David_Chang_David_Shankbone_2010.jpg"
]

interface = AnnotatorInterFace.for_image_generation_preference(
    prompts=prompts,
    completions_a=images_a,
    completions_b=images_b,
    fn_model=None, # a callable e.g. (function or transformers pipelines) that returns `str`
    fn_next_input=None, # a function that feeds gradio components actively with the next input
    csv_logger=False, # True if you want to log to a CSV
    dataset_name=None # "<my_hf_org>/<my_dataset>" if you want to log to the hub
)
interface.launch()
```

</details>

### Synthesizer

> Built on top of the `distilabel` to synthesize data with models in the loop.

> [!TIP]
> You can use also call the synthesizer directly to generate data. `synthesizer() -> Tuple` or `Synthesizer.batch_synthesize(n) -> List[Tuple]` to get inputs for the various tasks.

<details>
<summary><code>text-classification</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_text_classification(prompt_context="IMDB movie reviews")

interface = AnnotatorInterFace.for_text_classification(
    fn_next_input=synthesizer,
    labels=["positive", "negative"]
)
interface.launch()
```

</details>

<details>
<summary><code>text-generation</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_text_generation(
    prompt_context="A phone company customer support expert"
)

interface = AnnotatorInterFace.for_text_generation(
    fn_next_input=synthesizer
)
interface.launch()
```

</details>

<details>
<summary><code>chat-classification</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_chat_classification(
    prompt_context="A phone company customer support expert"
)

interface = AnnotatorInterFace.for_chat_classification(
    fn_next_input=synthesizer,
    labels=["positive", "negative"]
)
interface.launch()
```

</details>

<details>
<summary><code>chat-generation</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_chat_generation(
    prompt_context="A phone company customer support expert"
)

interface = AnnotatorInterFace.for_chat_generation(
    fn_next_input=synthesizer
)
interface.launch()
```

</details>

<details>
<summary><code>chat-generation-preference</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_chat_generation_preference(prompt_context="A phone company customer support expert")

interface = AnnotatorInterFace.for_chat_generation_preference(
    fn_next_input=synthesizer
)
interface.launch()
```

</details>

<details>
<summary><code>image-classification</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_image_classification(prompt_context="A phone company customer support expert")

interface = AnnotatorInterFace.for_image_classification(
    fn_next_input=synthesizer,
    labels=["positive", "negative"]
)
interface.launch()
```

</details>

<details>

<summary><code>image-generation</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_image_generation(prompt_context="A phone company customer support expert")

interface = AnnotatorInterFace.for_image_generation(
    fn_next_input=synthesizer
)
interface.launch()
```

</details>

<details>

<summary><code>image-description</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_image_description(prompt_context="A phone company customer support expert")

interface = AnnotatorInterFace.for_image_description(
    fn_next_input=synthesizer
)
interface.launch()
```

</details>

<details>

<summary><code>image-question-answering</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_image_question_answering(prompt_context="A phone company customer support expert")

interface = AnnotatorInterFace.for_image_question_answering(
    fn_next_input=synthesizer
)
interface.launch()
```

</details>

<details>

<summary><code>image-generation-preference</code></summary>

```python
from dataset_viber import AnnotatorInterFace
from dataset_viber.synthesizer import Synthesizer

synthesizer = Synthesizer.for_image_generation_preference(prompt_context="A phone company customer support expert")

interface = AnnotatorInterFace.for_image_generation_preference(
    fn_next_input=synthesizer
)
interface.launch()
```

</details>

### BulkInterface

> Built on top of the `Dash`, `plotly-express`, `umap-learn`, and `fast-sentence-transformers` to embed and understand your distribution and annotate your data.

https://github.com/user-attachments/assets/5e96c06d-e37f-45a0-9633-1a8e714d71ed

[Hub dataset](https://huggingface.co/datasets/SetFit/ag_news)

<details>
<summary><code>text-visualization</code></summary>

```python
from dataset_viber import BulkInterface
from datasets import load_dataset

ds = load_dataset("SetFit/ag_news", split="train[:2000]")

interface: BulkInterface = BulkInterface.for_text_visualization(
    ds.to_pandas()[["text", "label_text"]],
    content_column='text',
    label_column='label_text',
)
interface.launch()
```

</details>

<details>
<summary><code>text-classification</code></summary>

```python
from dataset_viber import BulkInterface
from datasets import load_dataset

ds = load_dataset("SetFit/ag_news", split="train[:2000]")
df = ds.to_pandas()[["text", "label_text"]]

interface = BulkInterface.for_text_classification(
    dataframe=df,
    content_column='text',
    label_column='label_text',
    labels=df['label_text'].unique().tolist()
)
interface.launch()
```

</details>

<details>
<summary><code>chat-visualization</code></summary>

```python
from dataset_viber.bulk import BulkInterface
from datasets import load_dataset

ds = load_dataset("argilla/distilabel-capybara-dpo-7k-binarized", split="train[:1000]")
df = ds.to_pandas()[["chosen"]]

interface = BulkInterface.for_chat_visualization(
    dataframe=df,
    chat_column='chosen',
)
interface.launch()
```

</details>

<details>
<summary><code>chat-classification</code></summary>

```python
from dataset_viber.bulk import BulkInterface
from datasets import load_dataset

ds = load_dataset("argilla/distilabel-capybara-dpo-7k-binarized", split="train[:1000]")
df = ds.to_pandas()[["chosen"]]

interface = BulkInterface.for_chat_classification(
    dataframe=df,
    chat_column='chosen',
    labels=["math", "science", "history", "question seeking"],
)
interface.launch()
```

</details>

### Utils

<details>
<summary>Shuffle inputs in the same order</summary>

When working with multiple inputs, you might want to shuffle them in the same order.

```python
def shuffle_lists(*lists):
    if not lists:
        return []

    # Get the length of the first list
    length = len(lists[0])

    # Check if all lists have the same length
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All input lists must have the same length")

    # Create a list of indices and shuffle it
    indices = list(range(length))
    random.shuffle(indices)

    # Reorder each list based on the shuffled indices
    return [
        [lst[i] for i in indices]
        for lst in lists
    ]
```

</details>

<details>
<summary>Random swap to randomize completions</summary>

When working with multiple completions, you might want to swap out the completions at the same index, where each completion index x is swapped with a random completion at the same index. This is useful for preference learning.

```python
def swap_completions(*lists):
    # Assuming all lists are of the same length
    length = len(lists[0])

    # Check if all lists have the same length
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All input lists must have the same length")

    # Convert the input lists (which are tuples) to a list of lists
    lists = [list(lst) for lst in lists]

    # Iterate over each index
    for i in range(length):
        # Get the elements at index i from all lists
        elements = [lst[i] for lst in lists]

        # Randomly shuffle the elements
        random.shuffle(elements)

        # Assign the shuffled elements back to the lists
        for j, lst in enumerate(lists):
            lst[i] = elements[j]

    return lists
```

</details>

<details>
<summary>Load remote image URLs from Hugging Face Hub</summary>

When working with images, you might want to load remote URLs from the Hugging Face Hub.

```python
from datasets import Dataset, Image, load_dataset

dataset = load_dataset(
    "my_hf_org/my_image_dataset"
).cast_column("my_image_column", Image(decode=False))
dataset[0]["my_image_column"]
# {'bytes': None, 'path': 'path_to_image.jpg'}
```

</details>

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

- <https://huggingface.co/spaces/davidberenstein1957/llm-human-feedback-collector-chat-interface-dpo>
- <https://huggingface.co/spaces/davidberenstein1957/llm-human-feedback-collector-chat-interface-kto>
- <https://medium.com/@oxenai/collecting-data-from-human-feedback-for-generative-ai-ec9e20bf01b9>
- <https://hamel.dev/notes/llm/finetuning/04_data_cleaning.html>
