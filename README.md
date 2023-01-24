# Function-Space Inference

## Setup

Create a new conda environment (if needed):
```
conda env create -f environment.yml -n <env_name>
```

Install CUDA-compiled PyTorch version from [here](https://pytorch.org). The codebase
has been tested with PyTorch version `1.13`.
```shell
pip install 'torch<1.14' torchvision --extra-index-url https://download.pytorch.org/whl/cu116
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

### Manual Changes

To load CIFAR-10 Corrupted configurations from TFDS, the following path is necessary in the `timm` library [parser_factory.py](https://github.com/rwightman/pytorch-image-models/blob/v0.6.7/timm/data/parsers/parser_factory.py#L9):

```diff
- name = name.split('/', 2)
+ name = name.split('/', 1)
```