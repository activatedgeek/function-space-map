# Function-Space Inference

This repository hosts the code for [Function-Space Regularization in Neural Networks: A Probabilistic Perspective]() (FSEB).


## Setup

Create a new conda environment (if needed):
```
conda env create -f environment.yml -n <env_name>
```

Install CPU-only PyTorch version from [here](https://pytorch.org) to avoid CUDA library conflicts. The codebase
has been tested with PyTorch version `1.13.1`.
```shell
pip install 'torch<2.0' torchvision --extra-index-url https://download.pytorch.org/whl/cpu/
```

Install CUDA-compiled JAX version from [here](https://github.com/google/jax#installation). The
codebase has been tested with JAX version `0.4`.
```shell
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

And finally, run
```
pip install -e .
```

## Usage

### Training PS-MAP

The main file for training `PS-MAP` is [experiments/train.py](./experiments/train.py).

An example command to run the training with FashionMNIST with `PS-MAP` is
```shell
python experiments/train.py \
    --dataset=fmnist --ood-dataset=mnist --batch-size=128 \
    --model-name=resnet18 \
     --lr=0.1 --weight-decay=5e-4 \
    --epochs=50 --seed=173
```

See below for a summary of all important arguments:
- `--dataset`: The training dataset. e.g. `fmnist` (FashionMNIST), `cifar10` (CIFAR-10), etc.
- `--ood-dataset`: The dataset used to evaluate OOD detection. e.g. `mnist` (MNIST), `svhn` (SVHN), etc.
- `--model-name`: We use `resnet18` (ResNet-18) for all our experiments.
- `--batch-size`: Size of the minibatch used for each gradient update.
- `--epochs`: Number of epochs to train for.
- `--lr`: Learning rate for SGD.
- `--weight-decay`: Weight decay for SGD.
- `--seed`: Seed used for model initialization, and dataset sampling.

In addition, there are a few more helpful arguments:
- `--log-dir`: Optional, but can be used to change the directory where checkpoints are stored. The full path is reported in the stdout for reference.

### Evaluate

The file [experiments/evaluate.py](./experiments/evaluate.py) can be used to evaluate trained checkpoints independently.

The key arguments are:
- `--model-name`:  We use `resnet18` (ResNet-18) for all our experiments, same as above.
- `--ckpt-path`: Path to the directory where the checkpoint is stored, or the checkpoint directly. This is often the prefix of the `--log-dir` reported in the training runs.
- `--dataset`: The dataset with which the checkpoint was trained (so that the checkpoint parameters can be mapped correctly).
- `--batch-size`: Batch size used during evaluation. A larger batch size can be used since we do not need memory to preserve computational graphs.

### Evaluate Landscape

The file [experiments/evaluate_landscape.py](./experiments/evaluate_landscape.py) can be used to assess the landscape around trained checkpoints.
We use [Filter Normalization](https://arxiv.org/abs/1712.09913) to plot perturbations along random directions around the optimum.

In terms of design, the number of GPUs avaiable decides the number of random directions we evaluation in a single run (via `pmap`),
and on each GPU we take multiple steps in the random directions evaluated (via `vmap`).

A sample command to evaluate CIFAR-10 landscape is:
```shell
python experiments/evaluate_landscape.py \
    --dataset=cifar10 --model-name=resnet18 \
    --batch-size=128 --ckpt-path=${CKPT_PATH} \
    --step_lim=50. --n-steps=50 \
    --seed=137
```

The new key arguments are:
- `--step-lim`: The limit of the step size `s`, such that we take steps along the random direction as `[-s, s]`.
- `--n-steps`: The number of steps in the range `[-s,s]`.

Use `CUDA_VISIBLE_DEVICES` to control the number of GPUs available for parallel evaluation of multiple random directions.
Alternatively, use multiple runs to get more random directions with different seeds.

## LICENSE

Apache 2.0
