<h1 align="center">Can You Hear Me Now?: A Benchmark for Long-Range Graph Propagation</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2512.17762">
        <img src="https://img.shields.io/badge/arXiv-2512.17762-b31b1b.svg" alt="arXiv">
    </a>
    <a href="https://huggingface.co/datasets/lucamiglior/echo-benchmark">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue" alt="Hugging Face Datasets">
    </a>
</p>

## Overview

**ECHO** is a novel benchmark designed to rigorously test the long-range information propagation capabilities of Graph Neural Networks (GNNs). While current benchmarks often focus on local interactions, ECHO introduces both synthetic and real-world tasks where successful prediction requires traversing up to **40 hops** in a graph.


## 🚀 Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

```bash
# Create a virtual environment
uv init
# Install dependencies
uv sync
```

### Downloading Data

You can download the datasets using the provided script. This will automatically fetch the required files from Hugging Face.

```bash
# Download all tasks
python scripts/download-all.py

# Or download specific tasks
python scripts/download-all.py --task diam
python scripts/download-all.py --task charge
```

---

## 📦 Datasets

### 🧪 ECHO-Synth
A synthetic dataset suite of **10,080 graphs** across six topologies: `line`, `ladder`, `grid-like`, `tree`, `caterpillar`, `lobster`.

Tasks:
- **Single-source shortest path (sssp)**
- **Node eccentricity (ecc)**
- **Graph diameter (diam)**

Each task isolates different aspects of global graph property prediction.

### 🧪 ECHO-Chem
A **real-world molecular dataset** of **200k molecular graphs**.

Tasks:
- **Atomic partial charges (charge)**: Node-level regression requiring long-range atomic interaction modeling.
- **Energy (energy)**: Graph-level regression task.

---

## 💻 Usage

### 📓 Notebooks (Recommended)
For ease of use, we provide Jupyter notebooks to guide you through training and inference:

- **`notebooks/train-model.ipynb`**: Step-by-step guide to training a model on ECHO tasks.
- **`notebooks/make-predictions.ipynb`**: Load a trained checkpoint and generate predictions.

### 🧠 Training from CLI
You can also train models directly using the command line script:

```bash
python scripts/train.py \
    --task diam \
    --gnn_type GNN \
    --conv_layer GCNConv \
    --hidden_dim 128 \
    --num_layers 8 \
    --lr 0.001 \
    --batch_size 512
```

### 🔎 Hyperparameter Search
To run a hyperparameter search (using Ray Tune and Optuna):

```bash
# Run search for diameter task
python scripts/search.py --tasks diam --n_samples 32

# Or use the helper script
bash scripts/run_search.sh
```

---


## 📚 Citation

If you use ECHO in your research, please cite our paper:

```bibtex
@article{echo2025benchmark,
    title={Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation}, 
    author={Luca Miglior and Matteo Tolloso and Alessio Gravina and Davide Bacciu},
    year={2025},
    journal={arXiv preprint arXiv:2512.17762}
}
```

