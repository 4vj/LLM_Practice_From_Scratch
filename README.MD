# LLM_PracticeV2

This repository is a personal learning project: experimenting with core building blocks of Large language models (attention, multi‑head attention, simple transformer blocks) implemented from scratch in NumPy and a tiny PyTorch variant

Status: experimental / learning

## Highlights
- [SimpleTransformer.py](SimpleTransformer.py) — tiny NumPy transformer class (`SimpleTransformer`) with a toy embedding dict and a `.generate()` demo.
- [PytorchTransformer.py](PytorchTransformer.py) — minimal PyTorch model (`MiniLLM`) and a small generate example.
- [Multiheaddemo.py](Multiheaddemo.py) — step‑by‑step multi‑head attention demo and visualization.
- [Bigattentiondemo.py](Bigattentiondemo.py) — attention heatmap demo.
- [attentiondemo.py](attentiondemo.py) — small attention score scratchpad.
- [softmax.py](softmax.py) — tiny softmax example.
- [Projectionvec.py](Projectionvec.py) and [Step1.py](Step1.py) — vector/projection plotting experiments.
- [Mission1.py](Mission1.py) — small PyTorch embedding lookup demo.
- [Test_Setup.py](Test_Setup.py) — prints the Python interpreter being used.

## Goals
- Understand attention and multi‑head attention mechanics visually and numerically.
- Build a tiny transformer from first principles (NumPy) and compare to a PyTorch implementation.
- Experiment with generation, temperature scaling, masking and layer normalization.
- understanding the math behind LLMS and building and training my own LLM.

## Requirements
- Python 3.8+
- numpy
- matplotlib
- torch (only needed for the PyTorch experiments)

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install numpy matplotlib torch
```

## Quick usage
Run the NumPy transformer demo:
```bash
python3 SimpleTransformer.py
```

Run the tiny PyTorch model demo:
```bash
python3 PytorchTransformer.py
```

View the attention demos:
```bash
python3 Multiheaddemo.py
python3 Bigattentiondemo.py
```