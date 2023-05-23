# Function-Space Inference

This repository holds the code to run experiments with L-MAP.

## Setup

Create a new conda environment (if needed):
```
conda env create -f environment.yml -n <env_name>
```

Install CUDA-compiled PyTorch version from [here](https://pytorch.org). The codebase
has been tested with PyTorch version `1.13`.
```shell
pip install 'torch<2.0' torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

Install CUDA-compiled JAX version from [here](https://github.com/google/jax#installation). The
codebase has been tested with JAX version `0.4`.
```shell
pip install 'jax[cuda11_cudnn82]<0.5' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

And finally, run
```
pip install -e .
```

## Usage

### Training L-MAP

The main file for training `L-MAP` is [experiments/train_lmap.py](./experiments/train_lmap.py).

An example command to run the training with FashionMNIST with `L-MAP` is
```shell
python experiments/train_lmap.py \
    --dataset=fmnist --ctx-dataset=kmnist \
    --model-name=resnet18 \
    --batch-size=128 --epochs=50 --lr=0.1 --weight-decay=5e-4 \
    --laplace-std=0.001 --reg_scale=5e-6 \
    --seed=173
```

See below for a small description of all important arguments:
- `--dataset`: The training dataset. e.g. `fmnist` (FashionMNIST), `cifar10` (CIFAR-10), etc.
- `--ctx-dataset`: The dataset used for evaluation points. e.g. `kmnist` (KMNIST), `whitenoise` (for White Noise distribution), etc.
- `--model-name`: We use `resnet18` (ResNet-18) for all our experiments.
- `--batch-size`: Size of the minibatch used for each gradient update.
- `--epochs`: Number of epochs to train for.
- `--lr`: Learning rate for SGD.
- `--weight-decay`: Weight decay for SGD.
- `--laplace-std`: The standard deviation used for the Laplacian estimator. `1e-3` is a good default, and mostly need not be changed.
- `--reg-scale`: The coefficient used for the Laplacian estimator. Use `0` to revert back to PS-MAP.
- `--seed`: Seed used for model initialization, and dataset sampling.

In addition, there are a few more helpful arguments:
- `--context-size`: Size of the sample from the evaluation dataset (`--ctx-dataset` above) to be used for Laplacian estimator. Defaults to `128`.
- `--label-noise`: A fraction `p` such that `p` fraction of classification labels are reassigned at random.
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
