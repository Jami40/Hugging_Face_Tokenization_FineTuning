# Hugging Face Basic Operations

An introductory Jupyter Notebook demonstrating common workflows with the
[Hugging Face Transformers](https://huggingface.co/docs/transformers) and
[Datasets](https://huggingface.co/docs/datasets) libraries.

The notebook starts with ready-to-use inference pipelines and tokenization,
then progresses to fine-tuning a BERT model for binary sentiment
classification on the IMDb movie-review dataset.

## Contents

- [Overview](#overview)
- [Notebook workflow](#notebook-workflow)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Fine-tuning configuration](#fine-tuning-configuration)
- [Generated artifacts](#generated-artifacts)
- [Notes and limitations](#notes-and-limitations)
- [References](#references)

## Overview

The notebook, [`Hugging_Face_Basic_Operation.ipynb`](./Hugging_Face_Basic_Operation.ipynb),
provides examples of:

- Loading and using `transformers.pipeline` for common machine-learning tasks.
- Running sentiment analysis on individual texts and batches of texts.
- Generating text with a pretrained language model.
- Answering questions from a supplied context.
- Loading custom pretrained models and tokenizers.
- Tokenizing, encoding, converting, and decoding text.
- Loading and preprocessing the Stanford IMDb dataset.
- Fine-tuning `bert-base-uncased` for two-class sequence classification.
- Evaluating and saving the fine-tuned model and tokenizer.

## Notebook workflow

### 1. Environment and installation

The first section checks NVIDIA GPU availability with `nvidia-smi` and
installs or upgrades the required Hugging Face packages:

- `transformers`
- `datasets`
- `huggingface_hub`

### 2. Transformers pipelines

The notebook introduces pipelines for several task families.

| Category | Examples covered |
| --- | --- |
| Natural language processing | Text classification, token classification, question answering, text generation, summarization, translation, text-to-text generation, fill-mask, feature extraction, and sentence similarity |
| Computer vision | Image classification, object detection, and image segmentation |
| Speech | Automatic speech recognition |
| Multimodal | Image-to-text |
| Structured/document tasks | Table question answering and document question answering |

Most examples allow Transformers to select a default model automatically.
The translation example explicitly uses
`Helsinki-NLP/opus-mt-en-fr`, while the text-generation example uses
`distilbert/distilgpt2`.

### 3. Sentiment analysis

The notebook demonstrates:

- Single-text sentiment classification.
- Batch inference over multiple texts.
- Replacing the default sentiment model with
  `SamLowe/roberta-base-go_emotions`.
- Loading a multilingual sentiment model,
  `nlptown/bert-base-multilingual-uncased-sentiment`, together with its
  tokenizer.

### 4. Text generation and question answering

The text-generation example prompts `distilgpt2` with:

```text
Today is a rainy day in London
```

The question-answering example uses a question and a context passage to
extract an answer.

### 5. Tokenization

Using `bert-base-uncased` and `bert-base-cased`, the notebook shows how to:

1. Split text into tokens.
2. Convert tokens into input IDs.
3. Build encoded model inputs, including attention masks.
4. Decode token IDs back into text.

### 6. IMDb fine-tuning

The final section fine-tunes `bert-base-uncased` on the
[`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)
dataset:

1. Download the dataset with `load_dataset`.
2. Tokenize review text with the BERT tokenizer.
3. Configure the Hugging Face `Trainer`.
4. Train for one epoch.
5. Evaluate on the test split.
6. Save the trained model and tokenizer locally.

The recorded notebook run produced an evaluation loss of approximately
`0.1796`. Results can vary depending on library versions, hardware, random
seeds, and training configuration.

## Requirements

- Python 3.9 or newer
- Jupyter Notebook, JupyterLab, or Google Colab
- Internet access to download models and datasets from the Hugging Face Hub
- Recommended: an NVIDIA GPU with a compatible CUDA-enabled PyTorch
  installation for faster fine-tuning

The notebook was originally run in a GPU-enabled Colab environment. CPU
execution is possible for the inference examples, but fine-tuning may be
significantly slower.

## Getting started

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Select **Runtime > Change runtime type**.
3. Choose a GPU accelerator if available.
4. Run the cells from top to bottom.

### Option 2: Local environment

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter transformers datasets huggingface_hub torch
```

Start Jupyter:

```bash
jupyter notebook
```

Then open [`Hugging_Face_Basic_Operation.ipynb`](./Hugging_Face_Basic_Operation.ipynb)
and execute the cells sequentially.

### Optional: Hugging Face authentication

Public models and datasets can usually be downloaded without authentication.
However, an access token is recommended for higher Hub rate limits and for
private or gated repositories:

```bash
hf auth login
```

Follow the prompts and paste a Hugging Face access token with the minimum
permissions required for your use case.

## Fine-tuning configuration

The notebook uses the following training settings:

| Setting | Value |
| --- | --- |
| Base model | `bert-base-uncased` |
| Number of labels | `2` |
| Learning rate | `2e-5` |
| Training batch size | `16` |
| Evaluation batch size | `16` |
| Epochs | `1` |
| Weight decay | `0.01` |
| Evaluation schedule | Every epoch |
| Output directory | `./results` |

These settings are intended for demonstration. For a production-quality
experiment, add reproducibility controls, evaluation metrics such as
accuracy or F1, checkpoint management, and validation across multiple runs.

## Generated artifacts

After the final cells complete, the notebook saves:

```text
fine-tuned-model/
fine-tuned-tokenizer/
```

The `results/` directory may also contain Trainer outputs and checkpoints.
These generated directories are local runtime artifacts and are not included
in the repository by default.

To load the saved model for inference:

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="./fine-tuned-model",
    tokenizer="./fine-tuned-tokenizer",
)

print(classifier("This movie was excellent."))
```

## Notes and limitations

- The notebook contains educational examples rather than a reusable
  production application.
- When a pipeline is created without a model name, Transformers selects a
  task-specific default model. Pin explicit model names and revisions when
  reproducibility or production stability matters.
- Model and dataset downloads can require substantial disk space and network
  bandwidth.
- The notebook uses `padding="max_length"` and `truncation=True` for
  simplicity. Production workloads may benefit from dynamic padding and
  task-specific sequence-length analysis.
- The IMDb labels are binary sentiment labels. The fine-tuned model should
  not be assumed to generalize to other domains without additional
  evaluation.
- Review model licenses, dataset terms, hardware requirements, and intended
  use before deploying any downloaded model.

## References

- [Transformers documentation](https://huggingface.co/docs/transformers)
- [Transformers pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets)
- [`bert-base-uncased` model card](https://huggingface.co/google-bert/bert-base-uncased)
- [IMDb dataset card](https://huggingface.co/datasets/stanfordnlp/imdb)
