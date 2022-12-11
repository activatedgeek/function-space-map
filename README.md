# Function-Space Inference

## Setup

Create a new conda environment (if needed):
```
conda env create -f environment.yml -n <env_name>
```

Install CUDA-compiled PyTorch version from [here](https://pytorch.org). The codebase
has been tested with PyTorch version `1.12`.
```shell
pip install 'torch<1.13' torchvision --extra-index-url https://download.pytorch.org/whl/cu116
```

Install CUDA-compiled JAX version from [here](https://github.com/google/jax#installation). The
codebase has been tested with JAX version `0.3`.
```shell
pip install 'jax[cuda11_cudnn82]<0.4' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

And finally, run
```
pip install -e .
```
