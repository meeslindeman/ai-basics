Hello

## Setup

Use the provided environment files:
```bash
conda env create -f environment.yml
conda activate pytorch-base
```

> [!TIP]
>
> For macOS users run the following comand to avoid warnings with `torch vision`:

```bash
conda install -c conda-forge jpeg libpng
```

## Basic

The `basic` folder containes simple networks. To run, for instance, a classic MLP on MNIST:

```
python -m models.basic.train --model mlp --dataset mnist
```

