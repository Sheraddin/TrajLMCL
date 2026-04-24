# TrajLMCL — Trajectory Data Analysis with LLM

Trajectory analysis project using a GPT-2 backbone (via LoRA fine-tuning) for road-network data. The model is trained incrementally across temporal data slices using knowledge distillation.

## Project Structure

```text
TrajLMCL/
├── config/         # Experiment configurations (JSON)
├── sample/         # Dataset samples (HDF5) & Download link
├── model/          # GPT-2 + LoRA + ConvEmbedder implementation
├── pretrain/       # Training and Distillation logic
├── downstream/     # TTE, Destination Prediction, and Search heads
├── dataloader/     # Data loading utilities
├── data.py         # Preprocessing entry point
└── main.py         # Main training and evaluation script
```

## Setup

### 1. Environment

```bash
conda env create -f environment.yml
conda activate trajcl
```

### 2. Model Weights

Place the GPT-2 pre-trained weights (`pytorch_model.bin`, `config.json`, etc.) into the `params/gpt2/` directory.

### 3. Data

Download the `.h5` dataset files from the link provided in `sample/README.md` and place them in the `sample/` folder.

## Quick Start

### Step 1: Preprocess

Run the preprocessing for the desired dataset (e.g., Chengdu):

```bash
# General dataset
python data.py -n small_chengdu -t trip,odpois-3,destination,tte -i 0,1,2

# CL stages (D0 - D4)
for i in {0..4}; do
  python data.py -n "small_chengdu_D$i" -t trip,odpois-3,destination,tte -i 0,1,2
done
```

### Step 2: Train & Evaluate

Start the training pipeline:

```bash
# Chengdu
python main.py --config small_chengdu --cuda 0

# Xi'an
python main.py --config small_xian --cuda 0
```

The script will execute the full pipeline: **D0 (warm-up) → D1 → D2 → D3 → D4**, evaluating Travel Time Estimation (TTE), Destination Prediction (DP), and Similar Trajectory Search (STS) at each step.
