# Proposed Attack Implementation

This repository contains the complete attack implementation code, which consists of three main parts:

- **Part 1**: Warm-up
- **Part 2**: Reinforcement Learning  
- **Part 3**: Selected Search

## Setup and Execution Guide

### 1. Environment Setup

```bash
cd ./ms-swift
```

Install all ms-swift framework related packages. For detailed instructions, see `./pthieft/attack/ms-swift/README.md`.

### 2. Data Preparation

**Training Dataset**: We use the Lexica dataset from [Hugging Face](https://huggingface.co/datasets/vera365/lexica_dataset).

We provide scripts to process this dataset for direct use in our attack:

```bash
python ./pthieft/attack/data/process.py
python ./pthieft/attack/data/process_json.py
```

**Model**: We use Llava 1.5 version from [ModelScope](https://modelscope.cn/models/llava-hf/llava-1.5-7b-hf).

### 3. Training Execution

We use Slurm training cluster for training our attack model (Part 1 + Part 2). We provide cluster bash scripts:

#### Part 1: Warm-up
```bash
sbatch ./pthieft/attack/script/SFT.sh
```

#### Part 2: Reinforcement Learning
```bash
sbatch ./pthieft/attack/script/llava_grpo.sh
```

#### Part 3: Selected Search

First, sample using:
```bash
swift sample \
    --model ./checkpoint-58000-merged \
    --sampler_engine pt \
    --num_return_sequences 10 \
    --dataset ./data/lexica_dataset/test_sample.json
```

Then run:
```bash
python ./pthieft/attack/data/resample.py
```

### 4. Evaluation

```bash
python ./pthieft/attack/data/eval.py
```
