# Collection of AI basics

In this repo I store a collection of simple AI models. This is for my own reference and learning.

## Setup

Use the provided environment files:
```bash
conda env create -f environment.yml
conda activate pytorch-base
```

> [!TIP]
>
> For macOS users run the following comand to avoid warnings with `torchvision`:

```bash
conda install -c conda-forge jpeg libpng
```

## Basic

The `basic` folder containes simple networks. To run, for instance, a classic MLP on MNIST:

```
python -m models.basic.train --model mlp --dataset mnist
```

