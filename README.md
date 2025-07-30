# See, Localize and Verify: A GRPO-Powered Framework for Enhancing Factual Accuracy in Multimodal Models

## Quick Start

### Environment Setup

```bash
# using conda
conda env create -f environment.yml
# or conda with pip
conda create -n qwen python==3.10.0
conda activate qwen && pip install requirements.txt
```

### Prepare dataset

You can download the full dataset following the instruction on the [challenge site](https://mm-hall-fact.github.io/ACMMM2025/), in [google drive](https://drive.google.com/drive/folders/1jrOxkw4UIQHU7EaAqZFemcp5tXjqFx-Z).

After downloaded, your file structure should look like:

```bash
. # dataset root directory
├── FactChecking
│   ├── images
│   └── json
└── HallucinationDetection
    ├── images
    └── json
```

### Evalution

You can evaluate your model by `vllm` and our scripts.

```bash
# server your model by vllm
bash eval/vllm_serve.sh YOUR_MODEL_PATH YOUR_MODEL_NAME
```

```bash
# server your model by vllm
bash eval/evaluation.sh MODEL_NAME_TO_SAVE_RESULT MODEL_NAME_IN_VLLM DATASET_ROOT_DIR
```

### Training and evaluating our model

Our training code and model are being organized and will be released soon in this repo and on Hugging Face.
