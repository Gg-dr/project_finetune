# Simplified Fine-Tuning Lab: LoRA vs. QLoRA

This repository contains everything you need to run a hands-on lab comparing LoRA and QLoRA fine-tuning on `Qwen2.5-1.5B-Instruct` using the Dolly-15k dataset. It is specifically designed to run within the constraints of a free Google Colab T4 GPU (16 GB VRAM).

## Directory Structure
```
lora-lab/
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── configs/
│   ├── lora.yaml
│   └── qlora.yaml
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── merge_adapter.py
└── notebooks/
    └── analysis.ipynb
```

## Step-by-Step Guide for Google Colab

**Step 0: Setup Environment**
Open Google Colab, select a T4 GPU runtime, and clone/upload this repository.
```bash
pip install -r requirements.txt
```

**Step 1: Data Preparation**
Download the databricks-dolly-15k dataset and format it into the ChatML template expected by Qwen.
```bash
python src/prepare_data.py
```
*This will save train, val, and test splits into `./data/processed/`.*

**Step 2: Train with LoRA**
Train the model using standard 16-bit LoRA. Note the GPU memory usage (often printed in Colab resources or via `nvidia-smi`).
```bash
python src/train.py --config configs/lora.yaml
```

**Step 3: Train with QLoRA**
Train the model using 4-bit quantized LoRA (QLoRA). 
```bash
python src/train.py --config configs/qlora.yaml
```
*Notice how QLoRA requires significantly less VRAM at the cost of a slightly longer initialization and iteration time!*

**Step 4: Evaluate and Compare**
This script will load the base model, LoRA adapter, and QLoRA adapter sequentially, generate answers for a subset of the test set, compute ROUGE scores, and output a side-by-side comparison.
```bash
python src/evaluate.py
```
*Read the generated `results_comparison.md` to see how the models perform qualitatively!*

**Step 5: Merge and Export**
Pick your favorite adapter and merge it back into the base model so you can use it in production or export it to formats like GGUF for Ollama.
```bash
python src/merge_adapter.py --adapter_path ./models/qlora --output_path ./models/qlora_merged
```
