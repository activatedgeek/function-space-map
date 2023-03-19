## Standard libraries
import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any
from collections import defaultdict
import time
import tree
import random as random_py
import functools
from functools import partial
from copy import copy
from typing import (Any, Callable, Iterable, Optional, Tuple, Union, Dict)
import warnings
import h5py
import argparse
from tqdm.auto import tqdm
import json

## Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import jit
from jax import config
from jax.config import config
# config.update("jax_debug_nans", True)
# config.update('jax_platform_name', 'cpu')

## Flax
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze
from flax.linen.initializers import lecun_normal

## JAX addons
import optax
# import distrax
# import neural_tangents as nt
# import flaxmodels as fm
from flaxmodels.resnet import ops
from flaxmodels import utils

## Tensorflow
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
tfd = tfp.distributions

## PyTorch
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, MNIST, KMNIST, ImageNet

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import datasets as sklearn_datasets
import wandb

from pathlib import Path
from timm.data import create_dataset
from torch.utils.data import Dataset, random_split

## Convert from CxHxW to HxWxC for Flax.
chw2hwc_fn = lambda img: img.permute(1, 2, 0)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='fmnist')  # cifar10, cifar10-224, cifar100, fmnist, two-moons
parser.add_argument('--train_subset', type=float, default=1.)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--context_subset", type=float, default=1.)
parser.add_argument("--context_set_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument('--model_name', type=str, default='ResNet18')  # ResNet9, ResNet18, ResNet18-Pretrained, ResNet50-Pretrained
parser.add_argument('--method', type=str, default='fsmap')  # fsmap, psmap, fsvi, psvi
parser.add_argument('--reg_type', type=str, default='function_prior')  # function_prior, function_norm, parameter_norm, feature_parameter_norm, function_kl, parameter_kl
parser.add_argument('--forward_points', type=str, default='train')
parser.add_argument('--context_points', type=str, default='train')
parser.add_argument("--context_transform", type=bool, default=False)
parser.add_argument('--ood_points', type=str, default='')
parser.add_argument("--mc_samples_llk", type=int, default=1)
parser.add_argument("--mc_samples_reg", type=int, default=1)
parser.add_argument("--mc_samples_eval", type=int, default=1)
parser.add_argument("--reg_scale", type=float, default=1)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--prior_mean", type=float, default=0)
parser.add_argument("--prior_var", type=float, default=0)
parser.add_argument("--prior_params_var", type=float, default=1)
parser.add_argument("--init_logvar", type=float, default=-50)
parser.add_argument("--init_final_layer_bias_logvar", type=float, default=-50)
parser.add_argument("--prior_feature_logvar", type=float, default=-50)
parser.add_argument("--prior_precision", type=float, default=0)
parser.add_argument("--pretrained_prior", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--linearize", action="store_true", default=False)
parser.add_argument("--evaluate", action="store_true", default=False)
parser.add_argument("--full_eval", action="store_true", default=False)
parser.add_argument("--restore_checkpoint", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--debug_print", action="store_true", default=False)
parser.add_argument("--debug_psd", action="store_true", default=False)
parser.add_argument("--log_frequency", type=int, default=2)
parser.add_argument("--save_to_wandb", action="store_true", default=False)
parser.add_argument('--wandb_project', type=str, default='function-space_transfer-learning')
parser.add_argument('--wandb_account', type=str, default='tgjr-research')
parser.add_argument('--gpu_mem_frac', type=float, default=0)
parser.add_argument('--config', type=str, default='')
parser.add_argument('--config_id', type=int, default=0)

# cwd = "/scratch-ssd/timner/deployment/testing/projects/function-space-variational-inference"

args = parser.parse_args()
args_dict = vars(args)

config = args.config
config_id = args.config_id

if config != '':
    with open(config, 'r') as f:
        config_json = json.load(f)

    configurations = config_json['configurations']
    name = configurations[config_id]['name']
    id = configurations[config_id]['id']
    cwd = configurations[config_id]['cwd']
    parser_args_list = configurations[config_id]['args']
    env_args = configurations[config_id]['env']

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
        
    parser_args = {}

    for i in range(len(parser_args_list)):
        if parser_args_list[i].startswith('--'):
            key = parser_args_list[i][2:]
            value = parser_args_list[i+1]
            parser_args[key] = value

    print(f"\nConfig name: {name}")
    print(f"\nConfig id: {id}")
    print(f"\nEnvironment args:\n\n{env_args}")

    for key in parser_args:
        args_dict[key] = parser_args[key]

    for key in parser_args:
        if isinstance(parser_args[key], int):
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and parser_args[key].isnumeric():
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and is_float(parser_args[key]):
            args_dict[key] = float(parser_args[key])
        elif parser_args[key] == 'True' or parser_args[key] == 'False':
            args_dict[key] = True if parser_args[key] == 'True' else False

    for key in env_args:
        os.environ[key] = env_args[key]

dataset = args_dict["dataset"]
train_subset = args_dict["train_subset"]
batch_size = args_dict["batch_size"]
context_subset = args_dict["context_subset"]
context_set_size = args_dict["context_set_size"]
num_epochs = args_dict["num_epochs"]
learning_rate = args_dict["learning_rate"]
alpha = args_dict["alpha"]
momentum = args_dict["momentum"]
model_name = args_dict["model_name"]
method = args_dict["method"]
reg_type = args_dict["reg_type"]
weight_decay = args_dict["weight_decay"]
context_points = args_dict["context_points"]
forward_points = args_dict["forward_points"]
context_transform = args_dict["context_transform"]
ood_points = args_dict["ood_points"]
mc_samples_llk = args_dict["mc_samples_llk"]
mc_samples_reg = args_dict["mc_samples_reg"]
mc_samples_eval = args_dict["mc_samples_eval"]
reg_scale = args_dict["reg_scale"]
prior_mean = args_dict["prior_mean"]
prior_var = args_dict["prior_var"]
prior_params_var = args_dict["prior_params_var"]
init_logvar = args_dict["init_logvar"]
init_final_layer_bias_logvar = args_dict["init_final_layer_bias_logvar"]
prior_feature_logvar = args_dict["prior_feature_logvar"]
prior_precision = args_dict["prior_precision"]
pretrained_prior = args_dict["pretrained_prior"]
linearize = args_dict["linearize"]
seed = args_dict["seed"]
evaluate = args_dict["evaluate"]
full_eval = args_dict["full_eval"]
restore_checkpoint = args_dict["restore_checkpoint"]
debug = args_dict["debug"]
debug_print = args_dict["debug_print"]
debug_psd = args_dict["debug_psd"]
log_frequency = args_dict["log_frequency"]
save_to_wandb = args_dict["save_to_wandb"]
wandb_project = args_dict["wandb_project"]
wandb_account = args_dict["wandb_account"]
gpu_mem_frac = args_dict["gpu_mem_frac"]

print(f"\nParser args:\n\n{args_dict}")

# print(f"\nCWD: {cwd}")
# os.chdir(cwd)

# if gpu_mem_frac != 0:
#     os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_frac)

# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints"

# Seeding for random operations
print(f"\nSeed: {seed}")
main_rng = random.PRNGKey(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
random_py.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.random.manual_seed(seed)

rng_key = main_rng

if debug:
    from jax import config
    config.update('jax_disable_jit', True)

jitter = eps = 1e-6

print(f"\nDevice: {jax.devices()[0]}\n")


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.
  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263
  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins
  Returns:
    ece: Expected Calibration Error
    mce: Maximum Calibration Error
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, mce


@jax.jit
def accuracy(logits_or_p, Y):
    '''Compute accuracy
    Arguments:
        logits_or_p: (B, d)
        Y: (B,) integer labels.
    '''
    if len(Y) == 0:
        return 0.
    matches = jnp.argmax(logits_or_p, axis=-1) == Y
    return jnp.mean(matches)


@jax.jit
def categorical_nll(logits, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return optax.softmax_cross_entropy_with_integer_labels(logits, Y)


@jax.jit
def categorical_nll_with_softmax(p, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return - jnp.sum(jnp.log(p + 1e-10) * jax.nn.one_hot(Y, p.shape[-1]), axis=-1)


@jax.jit
def categorical_entropy(p):
    '''Entropy of categorical distribution.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''
    return - jnp.sum(p * jnp.log(p + eps), axis=-1)


# @jax.jit
def selective_accuracy(p, Y):
    '''Selective Prediction Accuracy
    Uses predictive entropy with T thresholds.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''

    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores_id, threshold)
        mask = np.array(scores_id <= p)
        thresholded_accuracies.append(np.mean(accuracies_test[mask]))
    values_id = np.array(thresholded_accuracies)

    auc_sel_id = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel_id += (x * values_id[i] + x * values_id[i+1]) / 2

    return auc_sel_id


def selective_accuracy_test_ood(p_id, p_ood, Y):
    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p_id.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p_id)

    accuracies_ood = jnp.zeros(p_ood.shape[0])
    scores_ood = categorical_entropy(p_ood)

    accuracies = jnp.concatenate([accuracies_test, accuracies_ood], axis=0)
    scores = jnp.concatenate([scores_id, scores_ood], axis=0)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores, threshold)
        mask = np.array(scores <= p)
        thresholded_accuracies.append(np.mean(accuracies[mask]))
    values = np.array(thresholded_accuracies)

    auc_sel = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel += (x * values[i] + x * values[i+1]) / 2

    return auc_sel


def auroc_logits(predicted_logits_test, predicted_logits_ood, score, rng_key):
    predicted_targets_test = jax.nn.softmax(predicted_logits_test, axis=-1)
    predicted_targets_ood = jax.nn.softmax(predicted_logits_ood, axis=-1)

    ood_size = predicted_targets_ood.shape[1]
    test_size = predicted_targets_test.shape[1]
    anomaly_targets = jnp.concatenate((np.zeros(test_size), np.ones(ood_size)))
    if score == "entropy":
        entropy_test = categorical_entropy(predicted_targets_test.mean(0))
        entropy_ood = categorical_entropy(predicted_targets_ood.mean(0))
        scores = jnp.concatenate((entropy_test, entropy_ood))
    if score == "expected entropy":
        entropy_test = categorical_entropy(predicted_targets_test).mean(0)
        entropy_ood = categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((entropy_test, entropy_ood))
    elif score == "mutual information":
        mutual_information_test = categorical_entropy(predicted_targets_test.mean(0)) - categorical_entropy(predicted_targets_test).mean(0)
        mutual_information_ood = categorical_entropy(predicted_targets_ood.mean(0)) - categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((mutual_information_test, mutual_information_ood))
    else:
        NotImplementedError
    fpr, tpr, _ = roc_curve(anomaly_targets, scores)
    auroc_score = roc_auc_score(anomaly_targets, scores)
    return auroc_score


def merge_params(params_1, params_2):
    flat_params_1 = flax.traverse_util.flatten_dict(params_1)
    flat_params_2 = flax.traverse_util.flatten_dict(params_2)
    flat_params = flat_params_1 | flat_params_2
    unflat_params = flax.traverse_util.unflatten_dict(flat_params)
    return unflat_params


def split_params(params, type="dense"):
    flat_params_fixed = flax.traverse_util.flatten_dict(params)
    flat_params_rest = flax.traverse_util.flatten_dict(params)
    keys = flat_params_fixed.keys()

    i = -1
    for key in list(keys):
        if "Dense" in str(key) and "kernel" in str(key):
            i += 1

    if type == "dense":
        for key in list(keys):
            if f"Dense_{i}" in str(key):  # first check if there may be two final dense layers
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    elif type == "batch_norm":
        for key in list(keys):
            if "BatchNorm" in str(key):
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    else:
        raise NotImplementedError

    unflat_params_fixed = flax.traverse_util.unflatten_dict(flat_params_fixed)
    unflat_params_fixed = unflat_params_fixed
    unflat_params_rest = flax.traverse_util.unflatten_dict(flat_params_rest)
    unflat_params_rest = unflat_params_rest

    return unflat_params_fixed, unflat_params_rest


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class TwoMoons(torch.utils.data.Dataset):
    def __init__(self, size: int, inputs: np.ndarray, targets: np.ndarray):
        self.size = size
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input, target = self.inputs[index], int(self.targets[index][0].sum())
        ret = (input, target)
        return ret

def get_cifar10_test(root=None, v1=False, corr_config=None, batch_size=128, **_):
    _TEST_TRANSFORM = [
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4465), (.247, .243, .261)),
        transforms.Lambda(chw2hwc_fn)
    ]

    if v1:
        test_data = create_dataset(name='tfds/cifar10_1', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    elif corr_config is not None:
        test_data = create_dataset(f'tfds/cifar10_corrupted/{corr_config}', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    else:
        test_data = create_dataset('torch/cifar10', root=root, split='test',
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)

    return test_data

if dataset == 'cifar10' or dataset == 'cifar10-224':
    data_dir = os.environ.get('DATADIR')
    # _train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    _train_dataset = CIFAR10(root=data_dir, train=True, download=False)
    input_dim = 32
    num_classes = 10
    dataset_size = 50000
    testset_size = 10000
    training_dataset_size = 50000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size
    num_workers = 4
    persistent_workers = True
    
    # DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2))
    # DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2))
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.2023, 0.1994, 0.2010])
    DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    DATA_STD = np.array([0.247, 0.243, 0.261])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    test_transform_list = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]
    train_transform_list = [
        transforms.RandomCrop(input_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]

    if dataset == 'cifar10-224':
        test_transform_list.append(transforms.Resize(224))
        train_transform_list.append(transforms.Resize(224))

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    _train_dataset = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    # val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    val_dataset = CIFAR10(root=data_dir, train=True, transform=test_transform, download=False)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, 0], generator=torch.Generator().manual_seed(seed))
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [training_dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    # test_dataset = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    if context_transform:
        context_transform_list = [
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(32),
        ]
        if dataset == 'cifar10-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = CIFAR10(root=data_dir, train=True, transform=context_transform, download=False)
    elif context_points == "cifar100":
        context_dataset = CIFAR100(root=data_dir, train=True, transform=context_transform, download=False)
    elif context_points == "svhn":
        context_dataset = SVHN(root=Path(data_dir) / 'svhn', split="train", download=False, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root=data_dir, train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    nc = int(len(context_dataset) * np.abs(context_subset))
    context_set, _ = torch.utils.data.random_split(context_dataset, [nc, len(context_dataset) - nc], generator=torch.Generator().manual_seed(seed))

    if ood_points == "svhn":
        ood_dataset = SVHN(root=Path(data_dir) / 'svhn', split="test", download=False, transform=test_transform)
    elif ood_points == "cifar100":
        ood_dataset = CIFAR100(root=data_dir, train=False, download=False, transform=test_transform)
    else:
        ValueError("Unknown OOD dataset")

    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=4,
                                 persistent_workers=True)
    

    cifar101test_data = get_cifar10_test(root=data_dir, seed=seed, v1=True, corr_config=None, batch_size=batch_size_test)

    cifar101test_loader  = data.DataLoader(cifar101test_data,
                                batch_size=batch_size_test,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers,
                                persistent_workers=persistent_workers
                                )

    corr_config_list = [
        "speckle_noise_1", "speckle_noise_2", "speckle_noise_3", "speckle_noise_4", "speckle_noise_5",
        "shot_noise_1", "shot_noise_2", "shot_noise_3", "shot_noise_4", "shot_noise_5",
        "pixelate_1", "pixelate_2", "pixelate_3", "pixelate_4", "pixelate_5",
        "gaussian_blur_1", "gaussian_blur_2", "gaussian_blur_3", "gaussian_blur_4", "gaussian_blur_5",
        ]
    ccifar10test_loader_list = []
    # for corr_config in corr_config_list:
    #     ccifar10test_data = get_cifar10_test(root=data_dir, seed=seed, v1=False, corr_config=corr_config, batch_size=batch_size_test)

    #     ccifar10test_loader  = data.DataLoader(ccifar10test_data,
    #                                 batch_size=batch_size_test,
    #                                 shuffle=False,
    #                                 drop_last=False,
    #                                 num_workers=num_workers,
    #                                 persistent_workers=persistent_workers
    #                                 )
    #     ccifar10test_loader_list.append(ccifar10test_loader)

elif dataset == 'cifar100' or dataset == 'cifar100-224':
    # _train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    _train_dataset = CIFAR100(root="./data/CIFAR100", train=True, download=False)
    input_dim = 32
    num_classes = 100
    dataset_size = 50000
    testset_size = 10000
    training_dataset_size = 50000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size
    num_workers = 4
    persistent_workers = True

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2))
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.2023, 0.1994, 0.2010])
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.247, 0.243, 0.261])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    test_transform_list = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]
    train_transform_list = [
        transforms.RandomCrop(input_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]

    if dataset == 'cifar100-224':
        test_transform_list.append(transforms.Resize(224))
        train_transform_list.append(transforms.Resize(224))

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    _train_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=train_transform, download=False)
    # val_dataset = CIFAR100(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    val_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=test_transform, download=False)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, 0], generator=torch.Generator().manual_seed(seed))
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [training_dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = CIFAR100(root="./data/CIFAR100", train=False, transform=test_transform, download=False)

    if context_transform:
        context_transform_list = [
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(32),
        ]
        if dataset == 'cifar100-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=context_transform, download=False)
    elif context_points == "cifar100":
        context_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=context_transform, download=False)
    elif context_points == "svhn":
        context_dataset = SVHN(root="./data/SVHN", split="train", download=False, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    context_set, _ = torch.utils.data.random_split(context_dataset, [50000, 0], generator=torch.Generator().manual_seed(seed))

    if ood_points == "svhn":
        ood_dataset = SVHN(root="./data/SVHN", split="test", download=False, transform=test_transform)
    elif ood_points == "cifar100":
        ood_dataset = CIFAR100(root="./data/CIFAR100", train=False, download=False, transform=test_transform)
    else:
        ValueError("Unknown OOD dataset")

    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=4,
                                 persistent_workers=True)

elif dataset == 'fmnist' or dataset == 'fmnist-224':
    data_dir = os.environ.get('DATADIR')
    _train_dataset = FashionMNIST(root=data_dir, train=True, download=False)
    input_dim = 28
    num_classes = 10
    dataset_size = 60000
    testset_size = 10000
    training_dataset_size = 60000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size
    num_workers = 4
    persistent_workers = True

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2)).numpy()
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2)).numpy()
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)[:,:,None]
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img


    test_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    train_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    if dataset == 'fmnist-224':
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img

        test_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]
        train_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]

        test_transform_list.append(image_to_numpy)
        train_transform_list.append(image_to_numpy)

    _train_dataset = FashionMNIST(root=data_dir, train=True, transform=train_transform, download=False)
    val_dataset = FashionMNIST(root=data_dir, train=True, transform=test_transform, download=False)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, 0], generator=torch.Generator().manual_seed(seed))
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [training_dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = FashionMNIST(root=data_dir, train=False, transform=test_transform, download=False)

    if context_transform:
        context_transform_list = [
            transforms.Grayscale(1),
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(28),
        ]
        if dataset == 'fmnist-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = FashionMNIST(root=data_dir, train=True, transform=context_transform, download=False)
    elif context_points == "kmnist":
        context_dataset = KMNIST(root=data_dir, train=True, transform=context_transform, download=False)
    elif context_points == "mnist":
        context_dataset = MNIST(root=data_dir, train=True, download=False, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root=data_dir, train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    nc = int(len(context_dataset) * np.abs(context_subset))
    context_set, _ = torch.utils.data.random_split(context_dataset, [nc, len(context_dataset) - nc], generator=torch.Generator().manual_seed(seed))

    if ood_points == "mnist":
        ood_dataset = MNIST(root=data_dir, train=False, download=False, transform=test_transform)
    elif ood_points == "kmnist":
        ood_dataset = KMNIST(root=data_dir, train=False, transform=test_transform, download=False)
    else:
        ValueError("Unknown OOD dataset")

    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=4,
                                 persistent_workers=True)

elif dataset == 'mnist' or dataset == 'mnist-224':
    _train_dataset = FashionMNIST(root="./data/", train=True, download=False)
    input_dim = 28
    num_classes = 10
    dataset_size = 60000
    testset_size = 10000
    training_dataset_size = 60000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size
    num_workers = 4
    persistent_workers = True

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2)).numpy()
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2)).numpy()
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)[:,:,None]
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img


    test_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    train_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    if dataset == 'mnist-224':
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img

        test_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]
        train_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]

        test_transform_list.append(image_to_numpy)
        train_transform_list.append(image_to_numpy)

    _train_dataset = MNIST(root="./data/", train=True, transform=train_transform, download=False)
    val_dataset = MNIST(root="./data/", train=True, transform=test_transform, download=False)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, 0], generator=torch.Generator().manual_seed(seed))
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [training_dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = MNIST(root="./data/", train=False, transform=test_transform, download=False)

    if context_transform:
        context_transform_list = [
            transforms.Grayscale(1),
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(28),
        ]
        if dataset == 'mnist-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = MNIST("./data/", train=True, download=False, transform=context_transform)
    elif context_points == "kmnist":
        context_dataset = KMNIST(root="./data/", train=True, transform=context_transform, download=False)
    elif context_points == "fmnist":
        context_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=context_transform, download=False)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    context_set, _ = torch.utils.data.random_split(context_dataset, [60000, 0], generator=torch.Generator().manual_seed(seed))

    if ood_points == "fmnist":
        ood_dataset = FashionMNIST(root="./data/fashionMNIST", train=False, transform=test_transform, download=False)
    elif ood_points == "kmnist":
        ood_dataset = KMNIST(root="./data/", train=False, transform=test_transform, download=False)
    else:
        ValueError("Unknown OOD dataset")

    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=4,
                                 persistent_workers=True)

elif dataset == 'two-moons':
    input_dim = 2
    num_classes = 2
    dataset_size = 200
    training_dataset_size = 200
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = 32
    num_workers = 0
    persistent_workers = False

    x_train, y_train = sklearn_datasets.make_moons(
        n_samples=training_dataset_size, shuffle=True, noise=0.2, random_state=seed
    )
    y_train = np.array(y_train[:, None] == np.arange(2))

    context_lim = 2

    h_context = 0.25
    x_context_min, x_context_max = (
        x_train[:, 0].min() - context_lim,
        x_train[:, 0].max() + context_lim,
    )
    y_context_min, y_context_max = (
        x_train[:, 1].min() - context_lim,
        x_train[:, 1].max() + context_lim,
    )
    xx_context, yy_context = np.meshgrid(
        np.arange(x_context_min, x_context_max, h_context), np.arange(y_context_min, y_context_max, h_context)
    )
    x_context = np.vstack((xx_context.reshape(-1), yy_context.reshape(-1))).T
    y_context = jnp.ones((x_context.shape[0], 2))

    # x_context = jnp.repeat(x_train, 100, axis=0)
    # y_context = jnp.repeat(y_train, 100, axis=0)

    h = 0.25
    test_lim = 3
    # x_min, x_max = x_train[:, 0].min() - test_lim, x_train[:, 0].max() + test_lim
    # y_min, y_max = x_train[:, 1].min() - test_lim, x_train[:, 1].max() + test_lim
    x_min, x_max = (
        x_train[:, 0].min() - test_lim,
        x_train[:, 0].max() + test_lim,
        )
    y_min, y_max = (
        x_train[:, 1].min() - test_lim,
        x_train[:, 1].max() + test_lim,
        )
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
    _x_test = np.vstack((xx.reshape(-1), yy.reshape(-1))).T
    _y_test = jnp.ones((_x_test.shape[0], 2))

    permutation = np.random.permutation(_x_test.shape[0])
    x_test = _x_test[permutation]
    y_test = _y_test[permutation]

    permutation_inv = np.argsort(permutation)

    h_wide = 0.25
    test_lim_wide = 7
    x_wide_min, x_wide_max = (
        x_train[:, 0].min() - test_lim_wide,
        x_train[:, 0].max() + test_lim_wide,
    )
    y_wide_min, y_wide_max = (
        x_train[:, 1].min() - test_lim_wide,
        x_train[:, 1].max() + test_lim_wide,
    )
    xx_wide, yy_wide = np.meshgrid(
        np.arange(x_wide_min, x_wide_max, h_wide), np.arange(y_wide_min, y_wide_max, h_wide)
    )
    x_test_wide = np.vstack((xx_wide.reshape(-1), yy_wide.reshape(-1))).T
    y_test_wide = jnp.ones((x_test_wide.shape[0], 2))

    testset_size = x_test.shape[0]

    train_dataset = TwoMoons(x_train.shape[0], x_train, y_train)
    context_set = TwoMoons(x_context.shape[0], x_context, y_context)
    validation_dataset = TwoMoons(x_train.shape[0], x_train, y_train)
    test_dataset = TwoMoons(x_test.shape[0], x_test, y_test)

    ood_dataset = context_set
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=4,
                                 persistent_workers=True)
else:
    raise ValueError("Dataset not found.")
    

if np.abs(train_subset) < 1:
    n = len(train_dataset)
    ns = int(n * np.abs(train_subset))

    ## NOTE: -ve train_subset fraction to get latter segment.
    randperm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    randperm = randperm[ns:] if train_subset < 0 else randperm[:ns]

    train_dataset = data.Subset(train_dataset, randperm)

    print(f'Dataset size: {len(train_dataset)}')


train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate,
                               num_workers=8,
                               persistent_workers=True
                               )
context_loader  = data.DataLoader(context_set,
                               batch_size=context_set_size,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=num_workers,
                               persistent_workers=persistent_workers
                               )
val_loader   = data.DataLoader(validation_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=4,
                               persistent_workers=True
                               )
test_loader  = data.DataLoader(test_dataset,
                               batch_size=batch_size_test,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=num_workers,
                               persistent_workers=persistent_workers
                               )

class TrainState(train_state.TrainState):
    batch_stats: Any
    # params_logvar: Any


class TrainerModule:
    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 model_hparams : dict,
                 optimizer_name : str,
                 optimizer_hparams : dict,
                 objective_hparams : dict,
                 other_hparams: dict,
                 exmp_inputs : Any,
                 ):
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.objective_hparams = objective_hparams
        self.seed = other_hparams["seed"]
        self.num_epochs = other_hparams["num_epochs"]
        self.n_batches_eval = 10
        self.n_batches_eval_context = 10
        self.n_batches_eval_final = 100
        self.mc_samples_llk = objective_hparams["mc_samples_llk"]
        self.mc_samples_reg = objective_hparams["mc_samples_reg"]
        self.training_dataset_size = objective_hparams["training_dataset_size"]
        self.batch_size = objective_hparams["batch_size"]
        self.num_classes = self.model_hparams["num_classes"]
        self.linearize = self.model_hparams.pop('linearize', False)
        if model_name == 'ResNet18-Pretrained':
            self.model = ResNet18(output='logits', pretrained='imagenet', num_classes=self.num_classes, dtype='float32')
        elif model_name == 'ResNet50-Pretrained':
            self.model = ResNet50(output='logits', pretrained='imagenet', num_classes=self.num_classes, dtype='float32')
        else:
            self.model = self.model_class(**self.model_hparams)
        self.run_name = f"{method}_{reg_type}_{prior_var}_{prior_mean}_{reg_scale}_{learning_rate}_{alpha}_{num_epochs}_{context_points}_{model_name}_{dataset}_{seed}"
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.run_name)
        self.logger = {
            "epoch": [],
            "loss_train": [],
            "acc_train": [],
            "acc_test": [],
            "acc_test_best": [],
            "acc_sel_test": [],
            "acc_sel_test_ood": [],
            "nll_test": [],
            "ece_test": [],
            "ood_auroc_entropy": [],
            "ood_auroc_aleatoric": [],
            "ood_auroc_epistemic": [],
            "predictive_entropy_test": [],
            "aleatoric_uncertainty_test": [],
            "epistemic_uncertainty_test": [],
            "predictive_entropy_context": [],
            "aleatoric_uncertainty_context": [],
            "epistemic_uncertainty_context": [],
            "predictive_entropy_ood": [],
            "aleatoric_uncertainty_ood": [],
            "epistemic_uncertainty_ood": [],
        }
        if full_eval:
            if dataset == "cifar10":
                self.logger["acc_test_cifar101"] = []
                self.logger["acc_sel_test_cifar101"] = []
                self.logger["nll_test_cifar101"] = []
                self.logger["ece_test_cifar101"] = []
                for corr_config in corr_config_list:
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_sel_test_ccifar10_{corr_config}"] = []
        self.wandb_logger = []
        self.create_functions()
        self.prior_mean = objective_hparams["prior_mean"]
        self.prior_var = objective_hparams["prior_var"]
        self.init_logvar = objective_hparams["init_logvar"]
        self.init_final_layer_bias_logvar = objective_hparams["init_final_layer_bias_logvar"]
        self.prior_feature_logvar = objective_hparams["prior_feature_logvar"]
        self.pretrained_prior = objective_hparams["pretrained_prior"]
        self.params_prior_mean = None
        self.params_prior_logvar = None
        self.batch_stats_prior = None
        self.pred_fn = None
        self.stochastic = stochastic
        self.init_model(exmp_inputs)
        # print(self.model.tabulate(random.PRNGKey(0), x=exmp_inputs[0]))

        assert self.mc_samples_llk == 1 if not self.stochastic else True
        assert mc_samples_eval == 1 if not self.stochastic else True
        assert self.mc_samples_reg == 1 # if not ("fsmap" in method or "fsvi" in method) else True  # currently not implemented

    def create_functions(self):
        def calculate_cov(jac, logvar):
            var = jnp.exp(logvar)
            # jac has shape (batch_dim, output_dim, params_dims...)
            # jac_2D has shape (batch_dim * output_dim, nb_params)
            batch_dim, output_dim = jac.shape[:2]
            jac_2D = jnp.reshape(jac, (batch_dim * output_dim, -1))
            # sigma_flatten has shape (nb_params,) and will be broadcasted to the same shape as jac_2D
            sigma_flatten = jnp.reshape(var, (-1,))
            # jac_sigma_product has the same shape as jac_2D
            jac_sigma_product = jnp.multiply(jac_2D, sigma_flatten)
            cov = jnp.matmul(jac_sigma_product, jac_2D.T)
            cov = jnp.reshape(cov, (batch_dim, output_dim, batch_dim, output_dim))
            return cov

        def calculate_moments(params_mean, params_logvar, inputs, batch_stats, rng_key):
            ### Split both mean and logvar parameters
            params_feature_mean, params_final_layer_mean = split_params(params_mean, "dense")
            params_feature_logvar, params_final_layer_logvar = split_params(params_logvar, "dense")

            ### sample feature parameters and merge with final-layer mean parameters
            params_feature_sample = sample_parameters(params_feature_mean, params_feature_logvar, self.stochastic, rng_key)
            params_partial_sample = merge_params(params_feature_sample, params_final_layer_mean)

            ### Compute single-sample MC estimate of mean of logits
            _out = self.model.apply(
                {'params': params_partial_sample, 'batch_stats': batch_stats},
                inputs,
                train=True,
                mutable=['batch_stats']
                )
            out, _ = _out
            logits_mean = out

            ### Compute single-sample MC estimate of covariance of logits
            pred_fn = lambda final_layer_params: self.model.apply({'params': merge_params(params_feature_sample, final_layer_params), 'batch_stats': batch_stats}, inputs, train=True, mutable=['batch_stats'])
            jacobian = jax.jacobian(pred_fn)(params_final_layer_mean)[0]
            logits_cov = tree.map_structure(calculate_cov, jacobian, params_final_layer_logvar)
            logits_cov = jnp.stack(tree.flatten(logits_cov), axis=0).sum(axis=0)

            return logits_mean, logits_cov

        def calculate_function_kl(params_variational_mean, params_variational_logvar, inputs, batch_stats, rng_key):
            ### set prior batch stats
            batch_stats_prior = jax.lax.stop_gradient(batch_stats)
                
            ### set prior mean parameters
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"])
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.tree_map(lambda x, y: x + y, jax.lax.stop_gradient(params), jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"]))

            ### set parameter prior variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(self.prior_var)

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = split_params(params_prior_logvar_init, "dense")

            ### set feature and final-layer logvar parameters separately
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            ### merge logvar parameters
            params_prior_logvar = merge_params(params_feature_prior_logvar, params_final_layer_prior_logvar)

            logits_prior_mean, logits_prior_cov = calculate_moments(params_prior_mean, params_prior_logvar, inputs, batch_stats_prior, rng_key)
            logits_variational_mean, logits_variational_cov = calculate_moments(params_variational_mean, params_variational_logvar, inputs, batch_stats, rng_key)

            # if debug_print:
            #     jax.debug.print("\ncov prior:\n{}", logits_prior_cov[0:2, 0, 0:2, 0])
            #     jax.debug.print("cov variational:\n{}\n", logits_variational_cov[0:2, 0, 0:2, 0])
            #     jax.debug.print("cov prior inv:\n{}", jnp.linalg.inv(logits_prior_cov)[0:2, 0, 0:2, 0])
            #     jax.debug.print("cov variational inv:\n{}\n", jnp.linalg.inv(logits_variational_cov)[0:2, 0, 0:2, 0])

            kl = 0
            n_samples = logits_variational_mean.shape[0]
            n_output_dims = logits_variational_mean.shape[-1]
            cov_jitter = 1e-6
            for j in range(n_output_dims):
                _logits_prior_mean = logits_prior_mean[:, j].transpose()
                _logits_prior_cov = logits_prior_cov[:, j, :, j]  # + jnp.eye(n_samples) * cov_jitter

                _logits_variational_mean = logits_variational_mean[:, j].transpose()
                _logits_variational_cov = logits_variational_cov[:, j, :, j] + jnp.eye(n_samples) * cov_jitter

                # _logits_prior_mean = logits_prior_mean[:, j].transpose()
                # _logits_prior_cov = logits_prior_cov[:, j, :, j] + jnp.eye(n_samples) * cov_jitter
                # _logits_prior_mean = jnp.ones(n_samples) * 0
                # _logits_prior_cov = jnp.eye(n_samples) * final_layer_prior_var

                q = tfd.MultivariateNormalFullCovariance(
                    loc=_logits_variational_mean,
                    covariance_matrix=_logits_variational_cov,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                p = tfd.MultivariateNormalFullCovariance(
                    loc=_logits_prior_mean,
                    covariance_matrix=_logits_prior_cov,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                kl += tfd.kl_divergence(q, p, allow_nan_stats=False)

            return kl

        def calculate_function_prior_density(logits_reg, params, inputs, batch_stats, rng_key):
            ### set prior batch stats
            batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            # batch_stats_prior = self.batch_stats_prior

            ### set parameter prior mean
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"])
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.lax.stop_gradient(params)
            params_feature_prior_mean, params_final_layer_prior_mean = split_params(params_prior_mean, "dense")

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = split_params(params_prior_logvar_init, "dense")

            ### set feature parameter logvar and final-layer parameter variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(self.prior_var)
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            params_feature_prior_sample = sample_parameters(params_feature_prior_mean, params_feature_prior_logvar, self.stochastic, rng_key)
            # params_feature_prior_sample = params_feature_prior_mean
            # params_feature_prior_sample = sample_parameters(jax.tree_map(lambda x: x * 0, params_feature_prior_mean), params_feature_prior_logvar, self.stochastic, rng_key)  # use for non-init distribution feature variance

            params_prior_sample = merge_params(params_feature_prior_sample, params_final_layer_prior_mean)

            ### feature covariance (mostly the same as Jacobian covariance (does not include bias term), up to numerical errors)
            # _out = self.model.apply({'params': params_prior_sample, 'batch_stats': batch_stats_prior},
            #                         inputs,
            #                         train=True,
            #                         feature=True,
            #                         mutable=['batch_stats'])
            # out, _ = _out
            # logits_prior_mean, feature_prior = jax.lax.stop_gradient(out[0]), jax.lax.stop_gradient(out[1])

            # feature_dim = params_prior_sample["Dense_0"]["kernel"].shape[0]
            # final_layer_prior_var = jax.tree_map(lambda x: x * 0 + self.prior_var, jax.lax.stop_gradient(params_prior_sample["Dense_0"]["kernel"]))
            # diag_mat = jnp.diagflat(final_layer_prior_var).reshape(feature_dim,self.num_classes,feature_dim,self.num_classes).transpose(1, 3, 0, 2)
            # logits_prior_cov = jnp.matmul(jnp.matmul(feature_prior, diag_mat), feature_prior.T).transpose(2, 0, 3, 1)
            # logits_prior_cov += jnp.ones_like(logits_prior_cov) * self.prior_var  # add bias variance

            ### Jacobian covariance (mostly the same as feature covariance (does include bias term), up to numerical errors)
            _out = self.model.apply(
                {'params': params_prior_sample, 'batch_stats': batch_stats_prior},
                inputs,
                train=True,
                mutable=['batch_stats']
                )
            out, _ = _out
            logits_prior_mean = jax.lax.stop_gradient(out)

            pred_fn = lambda final_layer_params: self.model.apply(
                {'params': merge_params(params_feature_prior_sample, final_layer_params), 'batch_stats': batch_stats_prior},
                inputs,
                train=True,
                mutable=['batch_stats']
                )
            jacobian = jax.jacobian(pred_fn)(params_final_layer_prior_mean)[0]
            logits_prior_cov = tree.map_structure(calculate_cov, jacobian, params_final_layer_prior_logvar)
            logits_prior_cov = jnp.stack(tree.flatten(logits_prior_cov), axis=0).sum(axis=0)

            p = tfd.MultivariateNormalFullCovariance(
                loc=logits_prior_mean[:, 0],  # assumes the prior is identical across output dimensions
                covariance_matrix=logits_prior_cov[:, 0, :, 0],
                validate_args=False,
                allow_nan_stats=True,
            )
            log_density = jnp.sum(p.log_prob(logits_reg.T))

            reg = -log_density

            # log_density = 0
            # for j in range(self.num_classes):
            #     p = tfd.MultivariateNormalFullCovariance(
            #         loc=logits_prior_mean[:, j],
            #         covariance_matrix=logits_prior_cov[:, j, :, j],
            #         validate_args=False,
            #         allow_nan_stats=True,
            #     )
            #     log_density += p.log_prob(logits_reg[:, j])
            #     reg = -log_density

            # if debug_print:
            #     jax.debug.print("\nf - mean: {}", jnp.mean(logits_reg[:, 0] - logits_prior_mean[:, 0]))
            #     jax.debug.print("log_density: {}\n", p.log_prob(logits_reg.T))
            #     jax.debug.print("cholesky: {}\n", jnp.linalg.cholesky(logits_prior_cov[:, 0, :, 0]))

            # ### L2 regularization on non-final-layer parameters
            # params_model, _ = split_params(params, "dense")
            # reg += 1 / (2 * (1/0.025)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_model))[0])  # 1/2 * ||params - params_prior_mean||^2

            # ### L2 regularization on model parameters
            # params_model, _ = split_params(params, "batch_norm")
            # reg += 1 / (2 * (1/0.025)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_model))[0])  # 1/2 * ||params - params_prior_mean||^2

            # ### L2 regularization on BN parameters
            # _, params_batchnorm = split_params(params, "batch_norm")
            # reg += 1 / (2 * (1/0.007)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_batchnorm))[0])  # 1/2 * ||params - params_prior_mean||^2

            return reg

        def calculate_function_norm(logits_reg, inputs, batch_stats):
            if self.params_prior_mean is None:
                logits_prior_mean = jnp.zeros_like(logits_reg)
            else:
                _out = self.model.apply({'params': self.params_prior_mean, 'batch_stats': batch_stats},
                                        inputs,
                                        train=True,
                                        mutable=['batch_stats'])
                out, _ = _out
                logits_prior_mean = jax.lax.stop_gradient(out)
            reg = 1 / (2 * self.prior_var) * jnp.sum(jnp.square(logits_reg - logits_prior_mean))  # 1/(2 * function_var) * ||f(inputs, params) - f(inputs, params_prior_mean)||^2

            return reg

        def calculate_parameter_norm(params):
            params_model = params
            # params_model, params_batchnorm = split_params(params, "batch_norm")

            if self.objective_hparams["reg_type"] == "parameter_norm":
                params_reg = params_model
                if self.params_prior_mean is None:
                    params_reg_prior_mean = jax.tree_map(lambda x: x * 0, jax.lax.stop_gradient(params_model))
                else:
                    params_reg_prior_mean = self.params_prior_mean
                reg = 1 / (2 * self.prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params - params_prior_mean||^2
            elif self.objective_hparams["reg_type"] == "feature_parameter_norm":
                params_reg, _ = split_params(params_model, "dense")
                params_reg_prior_mean, _ = split_params(self.params_prior_mean, "dense")
                reg = 1 / (2 * self.prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params_feature - params_feature_prior_mean||^2
            else:
                raise NotImplementedError

            return reg

        def kl_univariate_gaussians(mean_q, var_q, mean_p, var_p):
            logstd_jitter = 0
            kl_1 = jnp.log((var_p + logstd_jitter) ** 0.5) - jnp.log((var_q + logstd_jitter) ** 0.5)
            kl_2 = ((var_q + logstd_jitter) + (mean_q - mean_p) ** 2) / (var_p + logstd_jitter)
            kl_3 = -1
            kl = 0.5 * (kl_1 + kl_2 + kl_3)
            return kl

        def calculate_parameter_kl(params_variational_mean, params_variational_logvar):
            if self.params_prior_mean is not None:
                params_prior_mean = self.params_prior_mean
                params_prior_logvar = self.params_prior_logvar
            else:
                params_prior_mean = jax.tree_map(lambda x: x * 0 + self.prior_mean, jax.lax.stop_gradient(params_variational_mean))
                params_prior_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(self.prior_var), jax.lax.stop_gradient(params_variational_logvar))

            ### remove batchnorm parameters from the KL calculation
            # params_prior_mean, _ = split_params(params_prior_mean, "batch_norm")
            # params_prior_logvar, _ = split_params(params_prior_logvar, "batch_norm")
            # params_variational_mean, _ = split_params(params_variational_mean, "batch_norm")
            # params_variational_logvar, _ = split_params(params_variational_logvar, "batch_norm")

            params_prior_var = jax.tree_map(lambda x: jnp.exp(x), params_prior_logvar)
            params_variational_var = jax.tree_map(lambda x: jnp.exp(x), params_variational_logvar)

            kl = 1 / (self.training_dataset_size / self.batch_size) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(
                lambda a, b, c, d: kl_univariate_gaussians(a, b, c, d),
                params_variational_mean, params_variational_var, params_prior_mean, params_prior_var
                ))[0])
            return kl

        def sample_parameters(params, params_logvar, stochastic, rng_key):
            if stochastic:
                eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), params_logvar)
                params_std_sample = jax.tree_map(lambda x, y: x * jnp.exp(y) ** 0.5, eps, params_logvar)
                params_sample = jax.tree_map(lambda x, y: x + y, params, params_std_sample)
            else:
                params_sample = params
            return params_sample
            
        def f_lin(params_dict, inputs, train, mutable):
            # params = params_dict["params"]
            # batch_stats = params_dict["batch_stats"]

            # out = self.model.apply(
            #     {'params': jax.lax.stop_gradient(params), 'batch_stats': batch_stats},
            #     inputs,
            #     train=train,
            #     mutable=['batch_stats'] if train else False
            #     )
            # _, new_model_state = out if train else (out, None)

            # batch_stats_linearization = new_model_state["batch_stats"] if train else batch_stats

            # if train:
            #     _pred_fn = lambda params: self.model.apply(
            #         {'params': params, 'batch_stats': batch_stats_linearization},
            #         inputs,
            #         train=True,
            #         mutable=['batch_stats']
            #         )[0]
            # else:
            #     _pred_fn = lambda params: self.model.apply(
            #         {'params': params, 'batch_stats': batch_stats_linearization},
            #         inputs,
            #         train=False,
            #         mutable=False
            #         )
            # pred_f, pred_jvp = jax.jvp(_pred_fn, (self.linearization_params["params"],), (jax.tree_map(lambda x, y: x - y, params, self.linearization_params["params"]),))
            # logits = pred_f + pred_jvp

            # if train:
            #     return logits, new_model_state
            # else:
            #     return logits

            params = params_dict["params"]
            batch_stats = params_dict["batch_stats"]

            out = self.model.apply(
                {'params': jax.lax.stop_gradient(params), 'batch_stats': batch_stats},
                inputs,
                train=train,
                mutable=['batch_stats'] if train else False
                )
            _, new_model_state = out if train else (out, None)

            if train:
                _pred_fn = lambda params: self.model.apply(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=True,
                    mutable=['batch_stats']
                    )[0]
        
                eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), jax.lax.stop_gradient(params))
                eps_scaled = jax.tree_map(lambda x, y: x * jnp.abs(y) * self.prior_var ** 0.5, eps, params)

                pred_f, pred_jvp = jax.jvp(_pred_fn, (params,), (jax.tree_map(lambda x: x, eps_scaled),))
                logits = pred_f + pred_jvp
            else:
                _pred_fn = lambda params: self.model.apply(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=False,
                    mutable=False
                    )
                pred_f = _pred_fn(params)
                logits = pred_f

            if train:
                return logits, new_model_state
            else:
                return logits

        def calculate_loss(params, params_logvar, rng_key, batch_stats, batch, batch_context, train):
            inputs, targets = batch
            _inputs_context, _ = batch_context

            if self.linearize and self.pred_fn is None:
                self.pred_fn = f_lin
            else:
                self.pred_fn = self.model.apply

            logits_list = []
            logits_reg_list = []
            for _ in range(self.mc_samples_llk):
                rng_key, _ = jax.random.split(rng_key)

                if self.objective_hparams["stochastic"]:
                    params = sample_parameters(params, params_logvar, self.stochastic, rng_key)

                if self.objective_hparams["forward_points"] == "train" or self.objective_hparams["forward_points"] == "none":
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    logits, new_model_state = out if train else (out, None)  # needed for cross-entropy loss

                    inputs_context = inputs  # needed for function_kl
                    logits_reg = logits  # needed for fsmap

                elif self.objective_hparams["forward_points"] == "joint":
                    inputs_joint = jnp.concatenate([inputs, _inputs_context], axis=0)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_joint,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    logits_joint, new_model_state = out if train else (out, None)
                    logits = logits_joint[:inputs.shape[0]]  # needed for cross-entropy loss

                    inputs_context = inputs_joint  # needed for function_kl and full_cov function_norm
                    logits_reg = logits_joint  # needed for fsmap
                elif self.objective_hparams["forward_points"] == "ood":
                    inputs_joint = jnp.concatenate([inputs, _inputs_context], axis=0)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_joint,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    logits_joint, new_model_state = out if train else (out, None)
                    logits = logits_joint[:inputs.shape[0]]  # needed for cross-entropy loss

                    inputs_context = _inputs_context  # needed for function_kl and full_cov function_norm
                    logits_reg = logits_joint  # needed for fsmap
                else:
                    raise ValueError("Unknown context points type.")
                    
                logits_list.append(logits)
                logits_reg_list.append(logits_reg)

            logits = jnp.stack(logits_list, axis=0)
            logits_reg = jnp.stack(logits_reg_list, axis=0)

            if self.objective_hparams["reg_type"] == "parameter_kl":
                assert self.objective_hparams["stochastic"] == True
                reg = calculate_parameter_kl(params, params_logvar)
            elif self.objective_hparams["reg_type"] == "function_kl":
                reg = calculate_function_kl(params, params_logvar, inputs_context, batch_stats, rng_key)
            elif self.objective_hparams["reg_type"] == "function_prior":
                reg = calculate_function_prior_density(logits_reg, params, inputs_context, batch_stats, rng_key)
            elif self.objective_hparams["reg_type"] == "function_norm":
                reg = calculate_function_norm(logits_reg, inputs_context, batch_stats)
            elif self.objective_hparams["reg_type"] == "parameter_norm":
                reg = calculate_parameter_norm(params)
            else:
                raise ValueError("Unknown regularization type.")

            # nll = categorical_nll_with_softmax(jax.nn.softmax(logits, -1), targets).mean(0).sum()
            # loss = (nll + self.objective_hparams["reg_scale"] * reg) / self.batch_size
            nll = jnp.stack([optax.softmax_cross_entropy_with_integer_labels(logits[i], targets) for i in range(self.mc_samples_llk)], axis=0).mean()
            loss = nll + self.objective_hparams["reg_scale"] * reg
            acc = 100 * (logits.argmax(axis=-1) == targets).mean()

            if debug_print:
                jax.debug.print("nll: {}", nll)
                jax.debug.print("reg: {}", reg)
                jax.debug.print("loss: {}", loss)
                jax.debug.print("acc: {}\n", acc)

            return loss, (acc, new_model_state)

        @jit
        def pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key):
            params = sample_parameters(params, params_logvar, self.stochastic, rng_key)
            logits = self.pred_fn(
                {'params': params, 'batch_stats': batch_stats},
                inputs,
                train=False,
                mutable=False
                )

            return logits

        def evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, type):
            if type == "test":
                _logits_test = []
                _targets_test = []

                for i, (batch, batch_2) in enumerate(zip(test_loader, context_loader)):
                    inputs_test, _targets = batch
                    inputs_context, _ = batch_2
                    n_context_points = inputs_context.shape[0]
                    inputs = jnp.concatenate([inputs_test, inputs_context], axis=0)
                    _logits_test_list = []
                    for _ in range(mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        pred = _pred[:_pred.shape[0] - n_context_points]
                        _logits_test_list.append(pred)
                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                    if i == n_batches_eval:
                        break
                logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                targets_test = jnp.concatenate(_targets_test, axis=0)

                ret = [logits_test, targets_test]

            elif type == "context":
                _logits_context = []
                for i, batch in enumerate(context_loader):
                    inputs_context, _ = batch
                    n_context_points = inputs_context.shape[0]
                    inputs = jnp.concatenate([inputs_context, inputs_context], axis=0)
                    _logits_context_list = []
                    for _ in range(mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        pred = _pred[:_pred.shape[0] - n_context_points]
                        _logits_context_list.append(pred)
                    _logits_context.append(jnp.stack(_logits_context_list, axis=0))
                    if i == self.n_batches_eval_context:
                        break
                logits_context = jnp.concatenate(_logits_context, axis=1)

                ret = [logits_context, None]

            elif type == "ood":
                _logits_ood = []
                for i, (batch, batch_2) in enumerate(zip(ood_loader, context_loader)):
                    inputs_ood, _ = batch
                    inputs_context, _ = batch_2
                    n_context_points = inputs_context.shape[0]
                    inputs = jnp.concatenate([inputs_ood, inputs_context], axis=0)
                    _logits_ood_list = []
                    for _ in range(mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        pred = _pred[:_pred.shape[0] - n_context_points]
                        _logits_ood_list.append(pred)
                    _logits_ood.append(jnp.stack(_logits_ood_list, axis=0))
                    if i == n_batches_eval:
                        break
                logits_ood = jnp.concatenate(_logits_ood, axis=1)

                ret = [logits_ood, None]

            if type == "cifar101":
                _logits_test = []
                _targets_test = []

                for i, (batch, batch_2) in enumerate(zip(cifar101test_loader, context_loader)):
                    inputs_test, _targets = batch
                    inputs_test, _targets = jnp.array(inputs_test), jnp.array(_targets)
                    inputs_context, _ = batch_2
                    n_context_points = inputs_context.shape[0]
                    inputs = jnp.concatenate([inputs_test, inputs_context], axis=0)
                    _logits_test_list = []
                    for _ in range(mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        pred = _pred[:_pred.shape[0] - n_context_points]
                        _logits_test_list.append(pred)
                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                    if i == n_batches_eval:
                        break
                logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                targets_test = jnp.concatenate(_targets_test, axis=0)

                ret = [logits_test, targets_test]

            if type == "corruptedcifar10":
                _logits_test_list_full = []
                _targets_test_list_full = []

                for ccifar10test_loader in ccifar10test_loader_list:
                    _logits_test = []
                    _targets_test = []
                    for i, (batch, batch_2) in enumerate(zip(ccifar10test_loader, context_loader)):
                        inputs_test, _targets = batch
                        inputs_test, _targets = jnp.array(inputs_test), jnp.array(_targets)
                        inputs_context, _ = batch_2
                        n_context_points = inputs_context.shape[0]
                        inputs = jnp.concatenate([inputs_test, inputs_context], axis=0)
                        _logits_test_list = []
                        for _ in range(mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                            pred = _pred[:_pred.shape[0] - n_context_points]
                            _logits_test_list.append(pred)
                        _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                        _targets_test.append(_targets)
                        if i == n_batches_eval:
                            break
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                    targets_test = jnp.concatenate(_targets_test, axis=0)[:testset_size]
                    _logits_test_list_full.append(logits_test)
                    _targets_test_list_full.append(targets_test)

                ret = [_logits_test_list_full, _targets_test_list_full]

            return ret

        def calculate_metrics(params, params_logvar, rng_key, batch_stats, n_batches_eval, full_eval):
            logits_test, targets_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "test")
            logits_context, _ = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "context")
            logits_ood, _ = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "ood")

            if full_eval:
                if dataset == "cifar10":
                    logits_cifar101, targets_cifar101 = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "cifar101")
                    logits_ccifar10_list, targets_ccifar10_list = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "corruptedcifar10")

                    acc_test_cifar101 = 100 * np.array(np.mean(jax.nn.softmax(logits_cifar101, axis=-1).mean(0).argmax(axis=-1) == targets_cifar101))
                    acc_sel_test_cifar101 = selective_accuracy(jax.nn.softmax(logits_cifar101, axis=-1).mean(0), targets_cifar101)
                    nll_test_cifar101 = jax.device_get(categorical_nll_with_softmax(jax.nn.softmax(logits_cifar101, -1).mean(0), targets_cifar101).mean())
                    ece_test_cifar101 = 100 * calibration(jax.nn.one_hot(targets_cifar101, self.num_classes), jax.nn.softmax(logits_cifar101, axis=-1).mean(0))[0]

                    self.logger["acc_test_cifar101"].append(acc_test_cifar101)
                    self.logger["acc_sel_test_cifar101"].append(acc_sel_test_cifar101)
                    self.logger["nll_test_cifar101"].append(nll_test_cifar101)
                    self.logger["ece_test_cifar101"].append(ece_test_cifar101)

                    for i, corr_config in enumerate(corr_config_list):
                        acc_test_ccifar10 = 100 * np.array(np.mean(jax.nn.softmax(logits_ccifar10_list[i], axis=-1).mean(0).argmax(axis=-1) == targets_ccifar10_list[i]))
                        acc_sel_test_ccifar10 = selective_accuracy(jax.nn.softmax(logits_ccifar10_list[i], axis=-1).mean(0), targets_ccifar10_list[i])
                        self.logger[f"acc_test_ccifar10_{corr_config}"].append(acc_test_ccifar10)
                        self.logger[f"acc_sel_test_ccifar10_{corr_config}"].append(acc_sel_test_ccifar10)

            acc_test = 100 * np.array(np.mean(jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1) == targets_test))
            acc_sel_test = selective_accuracy(jax.nn.softmax(logits_test, axis=-1).mean(0), targets_test)
            acc_sel_test_ood = selective_accuracy_test_ood(
                jax.nn.softmax(logits_test, axis=-1).mean(0),
                jax.nn.softmax(logits_ood, axis=-1).mean(0),
                targets_test
                )
            nll_test = jax.device_get(categorical_nll_with_softmax(jax.nn.softmax(logits_test, -1).mean(0), targets_test).mean())
            # nll_test = jax.device_get(categorical_nll(logits_test.mean(0), targets_test).mean())  # TODO: fix .mean(0) for BNNs
            ece_test = 100 * calibration(jax.nn.one_hot(targets_test, self.num_classes), jax.nn.softmax(logits_test, axis=-1).mean(0))[0]

            ood_auroc_entropy = 100 * auroc_logits(logits_test, logits_ood, score="entropy", rng_key=rng_key)
            ood_auroc_aleatoric = 100 * auroc_logits(logits_test, logits_ood, score="expected entropy", rng_key=rng_key)
            ood_auroc_epistemic = 100 * auroc_logits(logits_test, logits_ood, score="mutual information", rng_key=rng_key)

            predictive_entropy_test = jax.device_get(categorical_entropy(jax.nn.softmax(logits_test, -1).mean(0)).mean(0))
            predictive_entropy_context = jax.device_get(categorical_entropy(jax.nn.softmax(logits_context, -1).mean(0)).mean(0))
            predictive_entropy_ood = jax.device_get(categorical_entropy(jax.nn.softmax(logits_ood, -1).mean(0)).mean(0))
            aleatoric_uncertainty_test = jax.device_get(categorical_entropy(jax.nn.softmax(logits_test, -1)).mean(0).mean(0))
            aleatoric_uncertainty_context = jax.device_get(categorical_entropy(jax.nn.softmax(logits_context, -1)).mean(0).mean(0))
            aleatoric_uncertainty_ood = jax.device_get(categorical_entropy(jax.nn.softmax(logits_ood, -1)).mean(0).mean(0))
            epistemic_uncertainty_test = predictive_entropy_test - aleatoric_uncertainty_test
            epistemic_uncertainty_context = predictive_entropy_context - aleatoric_uncertainty_context
            epistemic_uncertainty_ood = predictive_entropy_ood - aleatoric_uncertainty_ood

            self.logger["acc_test"].append(acc_test)
            self.logger["acc_sel_test"].append(acc_sel_test)
            self.logger["acc_sel_test_ood"].append(acc_sel_test_ood)
            self.logger["nll_test"].append(nll_test)
            self.logger["ece_test"].append(ece_test)
            self.logger["ood_auroc_entropy"].append(ood_auroc_entropy)
            self.logger["ood_auroc_aleatoric"].append(ood_auroc_aleatoric)
            self.logger["ood_auroc_epistemic"].append(ood_auroc_epistemic)
            self.logger["predictive_entropy_test"].append(predictive_entropy_test)
            self.logger["predictive_entropy_context"].append(predictive_entropy_context)
            self.logger["predictive_entropy_ood"].append(predictive_entropy_ood)
            self.logger["aleatoric_uncertainty_test"].append(aleatoric_uncertainty_test)
            self.logger["aleatoric_uncertainty_context"].append(aleatoric_uncertainty_context)
            self.logger["aleatoric_uncertainty_ood"].append(aleatoric_uncertainty_ood)
            self.logger["epistemic_uncertainty_test"].append(epistemic_uncertainty_test)
            self.logger["epistemic_uncertainty_context"].append(epistemic_uncertainty_context)
            self.logger["epistemic_uncertainty_ood"].append(epistemic_uncertainty_ood)

        def train_step(state, batch, batch_context, rng_key):
            loss_fn = lambda params, params_logvar: calculate_loss(params, params_logvar, rng_key, state.batch_stats, batch, batch_context, train=True)
            # Get loss, gradients for loss, and other outputs of loss function
            ret, _grads = jax.value_and_grad(loss_fn, argnums=(0,1,), has_aux=True)(state.params["params"], state.params["params_logvar"])
            grads, grads_logvar = _grads[0], _grads[1]
            loss, acc, new_model_state = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=freeze({"params": grads, "params_logvar": grads_logvar}), batch_stats=new_model_state['batch_stats'])
            return state, loss, acc

        def eval_step(state, rng_key, n_batches_eval, full_eval):
            calculate_metrics(state.params["params"], state.params["params_logvar"], rng_key, state.batch_stats, n_batches_eval, full_eval)

        self.train_step = jax.jit(train_step)
        self.evaluation_predictions = evaluation_predictions
        self.eval_step = eval_step
        self.pred_fn_jit = pred_fn_jit

    def init_model(self, exmp_inputs):
        init_rng = jax.random.PRNGKey(self.seed)
        init_rng_logvar, _ = random.split(init_rng)

        variables = self.model.init(init_rng, exmp_inputs, train=True)
        variables_logvar = self.model.init(init_rng_logvar, exmp_inputs, train=True)

        init_params = variables['params']

        if self.stochastic:
            init_params_logvar = jax.tree_map(lambda x: x + self.init_logvar, variables_logvar['params'])
            init_params_feature_logvar, init_params_final_layer_logvar = split_params(init_params_logvar, "dense")
            init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 + self.init_logvar, init_params_final_layer_logvar)
            init_params_final_layer_logvar["Dense_0"]["bias"] = init_params_final_layer_logvar["Dense_0"]["bias"] * 0 + self.init_final_layer_bias_logvar
            # init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 - 15, init_params_final_layer_logvar)
            # init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 - 50, init_params_final_layer_logvar)
            init_params_logvar = merge_params(init_params_feature_logvar, init_params_final_layer_logvar)
        else:
            init_params_logvar = None

        # self.init_params = freeze({"params": init_params, "params_logvar": copy(init_params)})
        self.init_params = freeze({"params": init_params, "params_logvar": init_params_logvar})
        self.init_batch_stats = variables['batch_stats']

        self.linearization_params = jax.tree_map(lambda x: x * 1.00001, jax.lax.stop_gradient(self.init_params))
        self.linearization_batch_stats = jax.tree_map(lambda x: x * 1.00001, jax.lax.stop_gradient(self.init_batch_stats))

        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{self.optimizer_name}"'

        lr_schedule = optax.cosine_decay_schedule(
            init_value=self.optimizer_hparams.pop("lr"),
            decay_steps=num_steps_per_epoch*num_epochs,
            alpha=self.optimizer_hparams.pop("alpha"),
        )
        transf = []
        # transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **self.optimizer_hparams)
        )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
            )

    def train_model(self, train_loader, context_loader, val_loader, rng_key, num_epochs=200):
        print(f"\nTraining model for {num_epochs} epochs:\n")
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 0.0
        for epoch in tqdm(range(num_epochs), leave=False):
            epoch_idx = epoch + 1
            self.train_epoch(train_loader, context_loader, epoch=epoch_idx, rng_key=rng_key)
            if epoch_idx % log_frequency == 0:
                if dataset != "two-moons":
                    self.eval_model(rng_key, self.n_batches_eval)
                    if self.logger['acc_test'][-1] >= best_eval:
                        self.logger['acc_test_best'].append(self.logger['acc_test'][-1])
                        self.save_model(step=epoch_idx, best=True)
                    else:
                        self.logger['acc_test_best'].append(self.logger['acc_test_best'][-1])
                    best_eval = self.logger['acc_test_best'][-1]
                    
                    self.logger['epoch'].append(epoch_idx) 

                    if save_to_wandb and epoch_idx < num_epochs:
                        self.wandb_logger.append({})
                        for item in self.logger.items():
                            self.wandb_logger[-1][item[0]] = item[1][-1]
                        wandb.log(self.wandb_logger[-1])    

                    self.save_model(step=epoch_idx)

                    print(f"\nEpoch {epoch_idx}  |  Train Accuracy: {self.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {self.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {self.logger['ood_auroc_entropy'][-1]:.2f} / {self.logger['ood_auroc_aleatoric'][-1]:.2f} / {self.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f} / {self.logger['aleatoric_uncertainty_test'][-1]:.3f} / {self.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {self.logger['predictive_entropy_context'][-1]:.3f} / {self.logger['aleatoric_uncertainty_context'][-1]:.3f} / {self.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {self.logger['predictive_entropy_ood'][-1]:.3f} / {self.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {self.logger['epistemic_uncertainty_ood'][-1]:.3f}")
                
                else:  # two-moons                                     
                    _logits_sample_test = []
                    for i, (batch, batch_2) in enumerate(zip(test_loader, context_loader)):
                        inputs_test, _targets = batch
                        inputs_context, _ = batch_2
                        n_context_points = inputs_context.shape[0]
                        inputs = jnp.concatenate([inputs_test, inputs_context], axis=0)
                        _logits_sample_test_list = []
                        for _ in range(mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            sample = self.pred_fn_jit(self.state.params["params"], self.state.params["params_logvar"], self.state.batch_stats, inputs, n_context_points, rng_key)
                            _logits_sample_test_list.append(sample)
                        _logits_sample_test.append(jnp.stack(_logits_sample_test_list, axis=0))
                    logits_sample_test = jnp.concatenate(_logits_sample_test, axis=1)[:, :testset_size, :]

                    preds_y_mean = jnp.mean(jax.nn.softmax(logits_sample_test, -1), 0)
                    preds_y_mean = preds_y_mean[permutation_inv]
                    prediction_mean = preds_y_mean[:, 0].reshape(xx.shape)

                    preds_y_var = jnp.var(jax.nn.softmax(logits_sample_test, -1), 0)
                    preds_y_var = preds_y_var[permutation_inv]
                    prediction_var = preds_y_var[:, 0].reshape(xx.shape)

                    plt.figure(figsize=(10, 7))
                    cbar = plt.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
                    cb = plt.colorbar(cbar,)
                    cb.ax.set_ylabel(
                        "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
                        rotation=270,
                        labelpad=40,
                        size=30,
                    )
                    # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
                    cb.ax.tick_params(labelsize=30)
                    plt.scatter(
                        x_train[y_train[:, 0] == 0, 0],
                        x_train[y_train[:, 0] == 0, 1],
                        color="cornflowerblue",
                        edgecolors="black",
                    )
                    plt.scatter(
                        x_train[y_train[:, 0] == 1, 0],
                        x_train[y_train[:, 0] == 1, 1],
                        color="tomato",
                        edgecolors="black",
                    )
                    plt.tick_params(labelsize=30)
                    plt.savefig(
                        f"figures/two_moons/two_moons_gp_tdvi_predictive_mean_{epoch_idx}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

                    plt.figure(figsize=(10, 7))
                    cbar = plt.contourf(xx, yy, prediction_var, levels=20, cmap=cm.coolwarm)
                    cb = plt.colorbar(cbar,)
                    cb.ax.set_ylabel(
                        "$\mathbb{V}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
                        rotation=270,
                        labelpad=40,
                        size=30,
                    )
                    # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
                    cb.ax.tick_params(labelsize=30)
                    plt.scatter(
                        x_train[y_train[:, 0] == 0, 0],
                        x_train[y_train[:, 0] == 0, 1],
                        color="cornflowerblue",
                        edgecolors="black",
                    )
                    plt.scatter(
                        x_train[y_train[:, 0] == 1, 0],
                        x_train[y_train[:, 0] == 1, 1],
                        color="tomato",
                        edgecolors="black",
                    )
                    plt.tick_params(labelsize=30)
                    plt.savefig(
                        f"figures/two_moons/two_moons_gp_tdvi_predictive_var_{epoch_idx}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

    def train_epoch(self, train_loader, context_loader, epoch, rng_key):
        metrics = defaultdict(list)
        data_loader = tqdm(zip(train_loader, context_loader), leave=False)
        train_acc = 0
        elapsed = 0
        for batch, batch_context in data_loader:
            self.state, loss, acc = self.train_step(self.state, batch, batch_context, rng_key)
            rng_key, _ = jax.random.split(rng_key)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
            if (data_loader.format_dict["elapsed"] - elapsed) >= 0.5:  # update every 5 seconds
                train_acc = np.stack(jax.device_get(metrics["acc"]))[-40:].mean()  # average accuracy of last 40 batches
                data_loader.set_postfix({'accuracy': train_acc})
                elapsed = data_loader.format_dict["elapsed"]
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger[f"{key}_train"].append(avg_val)

    def eval_model(self, rng_key, n_batches_eval, full_eval=False):
        self.eval_step(self.state, rng_key, n_batches_eval, full_eval)

    def save_model(self, step=0, best=False):
        if best:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{best}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=0,
                overwrite=True,
                prefix=f"checkpoint_best_"
                )
            if save_to_wandb:
                wandb.save(f'{self.log_dir}/checkpoint_best_0')
        else:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{step}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=step,
                overwrite=True,
                prefix=f"checkpoint_"
                )
            if save_to_wandb:
                wandb.save(f'{self.log_dir}/checkpoint_{step}')

    def load_model(self, stochastic=False, pretrained_prior=False, restore_checkpoint=False):
        if not stochastic:
            if "Pretrained" in self.model_name:
                state_dict = self.model.init(rng_key, jnp.ones((1, 224, 224, 3)))
            else:
                # state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
                # ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/fspace-inference/wandb/fsmap_sanyam/checkpoint_30"  # fsmap fmnist OLD
                # ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/fspace-inference/wandb/fsmap_sanyam/checkpoint_198"  # psmap fmnist OLD
                # ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/fspace-inference/wandb/fsmap_sanyam/checkpoint_178"  # psmap cifar10 OLD
                # ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/function-space-variational-inference/checkpoints/checkpoint_fsmap_resnet18_cifar10_02"  # fsmap cifar10 OLD
                # ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/fspace-inference/wandb/fsmap_sanyam/cifar10_fsmap_fullcov"  # fsmap cifar10
                # ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/fspace-inference/wandb/fsmap_sanyam/fmnist_fsmap_fullcov"  # fsmap fmnist
                ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/fspace-inference/wandb/fsmap_sanyam/fmnist_fsmap_fullcov_02"  # fsmap fmnist
                state_dict = freeze(checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None))
            params_logvar = None
        else:
            if "Pretrained" in self.model_name and not restore_checkpoint:
                state_dict = self.model.init(rng_key, jnp.ones((1, 224, 224, 3)))
            else:
                # ckpt_path = "/scratch-ssd/timner/deployment/testing/projects/function-space-variational-inference/checkpoints/fsvi_function_kl_200_Pretrained Mean_1_0.005_0.05_50_train_ResNet18-Pretrained_cifar10-224_50/checkpoint_50"
                ckpt_path = f"/scratch-ssd/timner/deployment/testing/projects/function-space-variational-inference/checkpoints/fsvi_function_kl_10_0_1_0.005_0.05_200_cifar100_ResNet18_cifar10_{self.seed}_200/checkpoint_200"
                # ckpt_path = f"/scratch-ssd/timner/deployment/testing/projects/function-space-variational-inference/checkpoints/fsvi_function_kl_10_0_1_0.005_0.05_200_train_ResNet18_cifar10_{self.seed}_200/checkpoint_200"
                state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
                # state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
            
            params_logvar = state_dict['params_logvar']
            # params_logvar = jax.tree_map(lambda x: x + self.init_logvar, state_dict['params'])
            # params_feature_logvar, params_final_layer_logvar = split_params(params_logvar, "dense")
            # params_final_layer_logvar["Dense_0"]["bias"] = params_final_layer_logvar["Dense_0"]["bias"] * 0 + self.init_final_layer_bias_logvar
            # params_logvar = merge_params(params_feature_logvar, params_final_layer_logvar)

        if pretrained_prior:
            self.params_prior_mean = state_dict['params']
            self.batch_stats_prior = state_dict['batch_stats']
            self.params_prior_logvar = self.params_prior_logvar

        # self.batch_stats_prior = state_dict['batch_stats']  

        # if self.linearize:
        #     self.linearization_params = state_dict
        #     self.linearization_batch_stats = state_dict['batch_stats']

        if self.linearize and self.pred_fn is None:
            NotImplementedError
        else:
            self.pred_fn = self.model.apply

        params = freeze({"params": state_dict['params'], "params_logvar": params_logvar})

        self.state = TrainState.create(apply_fn=self.model.apply,
                                    params=params,
                                    # params_logvar=params_logvar,
                                    batch_stats=state_dict['batch_stats'],
                                    tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                    )

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))


def train_classifier(*args, rng_key, **kwargs):
    trainer = TrainerModule(*args, **kwargs)
    del kwargs['exmp_inputs']

    if save_to_wandb:
        wandb.config = copy(kwargs)
        wandb.init(
            mode=os.environ.get('WANDB_MODE', default='offline'),
            # project=wandb_project,
            # name=trainer.run_name,
            # entity=wandb_account,
            config=wandb.config,
        )
        trainer.log_dir = wandb.run.dir

    train = not evaluate
    
    if "Pretrained" in kwargs['model_name']:
        load_checkpoint = True
        prior = True
    else:
        # load_checkpoint = True
        load_checkpoint = False
        prior = False

    if train and not load_checkpoint:  # train from scratch
        trainer.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=trainer.num_epochs)
    elif train and load_checkpoint:  # load trained model and continue training
        trainer.load_model(stochastic=trainer.stochastic, pretrained_prior=trainer.pretrained_prior, restore_checkpoint=restore_checkpoint)
        trainer.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=trainer.num_epochs)
    else:  # load trained model and evaluate
        trainer.load_model(stochastic=trainer.stochastic, pretrained_prior=trainer.pretrained_prior, restore_checkpoint=restore_checkpoint)
        trainer.logger['acc_train'].append(0)
        trainer.logger['acc_test_best'].append(0)
        trainer.logger['loss_train'].append(0)
        trainer.logger['epoch'].append(trainer.num_epochs)

    if dataset != "two-moons":
        # val_acc = trainer.eval_model(val_loader, rng_key)
        trainer.eval_model(rng_key, trainer.n_batches_eval_final, full_eval=full_eval)
        # print(f"\nValidation Accuracy: {val_acc*100:.2f}")
        print(f"Train Accuracy: {trainer.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {trainer.logger['acc_test'][-1]:.2f}  |  Selective Accuracy: {trainer.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {trainer.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {trainer.logger['nll_test'][-1]:.3f}  |  Test ECE: {trainer.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {trainer.logger['ood_auroc_entropy'][-1]:.2f} / {trainer.logger['ood_auroc_aleatoric'][-1]:.2f} / {trainer.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {trainer.logger['predictive_entropy_test'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_test'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {trainer.logger['predictive_entropy_context'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_context'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {trainer.logger['predictive_entropy_ood'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_ood'][-1]:.3f}")
        
        if save_to_wandb:
            if evaluate:
                trainer.wandb_logger.append({})
                for item in trainer.logger.items():
                    trainer.wandb_logger[-1][item[0]] = item[1][-1]
                wandb.log(trainer.wandb_logger[-1])
                wandb.log(trainer.logger)
            else:
                trainer.wandb_logger.append({})
                for item in trainer.logger.items():
                    trainer.wandb_logger[-1][item[0]] = item[1][-1]
                wandb.log(trainer.wandb_logger[-1])
            time.sleep(10)
            wandb.finish()
    
    return trainer, trainer.logger


# conv_kernel_init = lecun_normal()  # flax default
conv_kernel_init = nn.initializers.variance_scaling(1/20, mode='fan_in', distribution='uniform')
# conv_kernel_init = nn.initializers.variance_scaling(1/20, mode='fan_in', distribution='normal')
# conv_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')


class CNN(nn.Module):
    """A simple CNN model."""
    num_classes : int
    act_fn : callable
    block_class : None
    num_blocks : None
    c_hidden : None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        # x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        # x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        # x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = x.reshape((x.shape[0], -1))  # flatten
        # x = nn.Dense(features=256, dtype=self.dtype)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        # _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)

        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        
        return x


class MLP(nn.Module):
    """A simple MLP model."""
    num_classes : int
    act_fn : callable
    block_class : None
    num_blocks : None
    c_hidden : None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True):
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.Dense(features=100, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=100, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        
        return x


class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=conv_kernel_init,
            use_bias=False
            )(x)
        z = nn.BatchNorm()(z, use_running_average=not train)

        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=conv_kernel_init,
            use_bias=False
            )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=conv_kernel_init
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=conv_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=conv_kernel_init,
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=conv_kernel_init,
                        use_bias=False)(x)

        x_out = z + x
        return x_out


class ResNetMod(nn.Module):
    num_classes : int
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(
            self.c_hidden[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            kernel_init=conv_kernel_init,
            use_bias=False
        )(x)
        # x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=conv_kernel_init, use_bias=False)(x)  # flax default
        if self.block_class == ResNetBlock:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample
                    )(x, train=train)

        # Mapping to classification output
        feature_map = x.mean(axis=(1, 2))

        x = nn.Dense(self.num_classes)(feature_map)

        if feature:
            return (x, feature_map)

        return x


URLS = {'resnet18': 'https://www.dropbox.com/s/wx3vt76s5gpdcw5/resnet18_weights.h5?dl=1',
        'resnet34': 'https://www.dropbox.com/s/rnqn2x6trnztg4c/resnet34_weights.h5?dl=1',
        'resnet50': 'https://www.dropbox.com/s/fcc8iii38ezvqog/resnet50_weights.h5?dl=1',
        'resnet101': 'https://www.dropbox.com/s/hgtnk586pnz0xug/resnet101_weights.h5?dl=1',
        'resnet152': 'https://www.dropbox.com/s/tvi28uwiy54mcfr/resnet152_weights.h5?dl=1'}

LAYERS = {'resnet18': [2, 2, 2, 2],
          'resnet34': [3, 4, 6, 3],
          'resnet50': [3, 4, 6, 3],
          'resnet101': [3, 4, 23, 3],
          'resnet152': [3, 8, 36, 3]}


class BasicBlock(nn.Module):
    """
    Basic Block.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: h5py.Group=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x 
        
        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(2, 2) if self.downsample else (1, 1),
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn2'],
                           dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features, 
                               kernel_size=(1, 1), 
                               strides=(2, 2), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = ops.batch_norm(residual,
                                      train=train,
                                      epsilon=1e-05,
                                      momentum=0.1,
                                      params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                                      dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        expansion (int): Factor to multiply number of output channels with.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: Any=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    expansion: int=4
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Bottleneck.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x 
        
        x = nn.Conv(features=self.features, 
                    kernel_size=(1, 1), 
                    strides=(1, 1),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=(3, 3), 
                    strides=(2, 2) if self.downsample and self.stride else (1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)
        
        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn2'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features * self.expansion, 
                    kernel_size=(1, 1), 
                    strides=(1, 1), 
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv3']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn3'],
                           dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features * self.expansion, 
                               kernel_size=(1, 1), 
                               strides=(2, 2) if self.stride else (1, 1), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = ops.batch_norm(residual,
                                      train=train,
                                      epsilon=1e-05,
                                      momentum=0.1,
                                      params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                                      dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class ResNet(nn.Module):
    """
    ResNet.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str): 
            Which ResNet model to use:
                - 'resnet18'
                - 'resnet34'
                - 'resnet50'
                - 'resnet101'
                - 'resnet152'
        num_classes (int):
            Number of classes.
        block (nn.Module):
            Type of residual block:
                - BasicBlock
                - Bottleneck
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """
    output: str='softmax'
    pretrained: str='imagenet'
    normalize: bool=True
    architecture: str='resnet18'
    num_classes: int=1000
    block: nn.Module=BasicBlock
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    ckpt_dir: str=None
    dtype: str='float32'

    def setup(self):
        self.param_dict = None
        if self.pretrained == 'imagenet':
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(ckpt_file, 'r')

    @nn.compact
    def __call__(self, x, train=True):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            If output == 'logits' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """
        if self.normalize:
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(self.dtype)  # EDITED
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(self.dtype)  # EDITED
            x = (x - mean) / std

        if self.pretrained == 'imagenet':
            # if self.num_classes != 1000: # EDITED
            #     warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} ' # EDITED
            #                     'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ', UserWarning) # EDITED
            # num_classes = 1000 # EDITED
            num_classes = self.num_classes # EDITED
        else:
            num_classes = self.num_classes
 
        act = {}

        x = nn.Conv(features=64, 
                    kernel_size=(7, 7),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']),
                    strides=(2, 2), 
                    padding=((3, 3), (3, 3)),
                    use_bias=False,
                    dtype=self.dtype)(x)
        act['conv1'] = x

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer 1
        down = self.block.__name__ == 'Bottleneck'
        for i in range(LAYERS[self.architecture][0]):
            params = None if self.param_dict is None else self.param_dict['layer1'][f'block{i}']
            x = self.block(features=64,
                           kernel_size=(3, 3),
                           downsample=i == 0 and down,
                           stride=i != 0,
                           param_dict=params,
                           block_name=f'block1_{i}',
                           dtype=self.dtype)(x, act, train)
        
        # Layer 2
        for i in range(LAYERS[self.architecture][1]):
            params = None if self.param_dict is None else self.param_dict['layer2'][f'block{i}']
            x = self.block(features=128,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block2_{i}',
                           dtype=self.dtype)(x, act, train)
        
        # Layer 3
        for i in range(LAYERS[self.architecture][2]):
            params = None if self.param_dict is None else self.param_dict['layer3'][f'block{i}']
            x = self.block(features=256,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block3_{i}',
                           dtype=self.dtype)(x, act, train)

        # Layer 4
        for i in range(LAYERS[self.architecture][3]):
            params = None if self.param_dict is None else self.param_dict['layer4'][f'block{i}']
            x = self.block(features=512,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block4_{i}',
                           dtype=self.dtype)(x, act, train)

        # Classifier
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=num_classes,
                     kernel_init=self.kernel_init if (self.param_dict is None or self.num_classes != 1000) else lambda *_ : jnp.array(self.param_dict['fc']['weight']),  # EDITED
                     bias_init=self.bias_init if (self.param_dict is None or self.num_classes != 1000)  else lambda *_ : jnp.array(self.param_dict['fc']['bias']),  # EDITED
                     dtype=self.dtype)(x)
        act['fc'] = x
        
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'activations':
            return act
        return x


def ResNet18(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet18 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet18',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet34(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet34 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet34',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet50(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet50 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet50',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet101(output='softmax',
              pretrained='imagenet',
              normalize=True,
              num_classes=1000,
              kernel_init=nn.initializers.lecun_normal(),
              bias_init=nn.initializers.zeros,
              ckpt_dir=None,
              dtype='float32'):
    """
    Implementation of the ResNet101 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet101',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet152(output='softmax',
              pretrained='imagenet',
              normalize=True,
              num_classes=1000,
              kernel_init=nn.initializers.lecun_normal(),
              bias_init=nn.initializers.zeros,
              ckpt_dir=None,
              dtype='float32'):
    """
    Implementation of the ResNet152 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet152',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


rng_key = main_rng

if debug:
    from jax import config
    config.update('jax_disable_jit', True)

if 'CNN' in model_name:
    model_class = CNN
    num_blocks = None
    c_hidden = None
if 'MLP' in model_name:
    model_class = MLP
    num_blocks = None
    c_hidden = None
if 'ResNet9' in model_name:  # 272,896 parameters for FMNIST
    model_class = ResNetMod
    num_blocks = (3, 3, 3)
    c_hidden = (16, 32, 64)
if 'ResNet18' in model_name:  # 11,174,642 parameters for FMNIST
    model_class = ResNetMod
    num_blocks = (2, 2, 2, 2)
    c_hidden = (64, 128, 256, 512)
if 'ResNet50' in model_name:
    model_class = ResNetMod
    num_blocks = None
    c_hidden = (64, 128, 256, 512)

block_class = ResNetBlock
# block_class = PreActResNetBlock

act_fn = nn.relu
# act_fn = nn.swish

if prior_precision == 0:
    prior_precision = 1 / prior_var
elif prior_var == 0:
    prior_var = 1 / prior_precision
else:
    raise ValueError("Only one of prior_precision and prior_var can be set.")

prior_mean = "Pretrained Mean" if "Pretrained" in model_name else prior_mean

### FS-MAP
if method == "fsmap":
    print(f"\n\n### FS-MAP ###\n")
    stochastic = False

    # learning_rate = 0.005  # CIFAR10
    # # learning_rate = 0.005  # FMNIST
    # # learning_rate = 0.01  # FMNIST
    # momentum = 0.9
    # alpha = 0.05  # CIFAR10-224 pretrained model
    # # alpha = 0.1  # CIFAR10
    # # alpha = 0.1  # FMNIST
    #
    # # reg_type = "function_norm"
    # reg_type = "function_prior"
    # 
    #
    # if reg_type == "function_norm":
    #     ### "function_norm"
    #     reg_scale = 1
    #     prior_precision = 0.00001  # CIFAR10
    #     context_points = "joint"  # CIFAR10
    #     weight_decay = 0  # CIFAR10
    #     # prior_precision = 0.00001  # FMNIST
    #     # context_points = "joint"  # FMNIST
    #     # weight_decay = 0  # FMNIST
    #     prior_var = 1 / prior_precision
    #
    # if reg_type == "function_prior":
    #     ### "function_prior"
    #     reg_scale = 1
    #     prior_mean = 0 if "Pretrained" not in model_name else "Pretrained Mean"
    #     prior_var = 5000  # CIFAR10
    #     # prior_var = 50000  # FMNIST
    #     context_points = "joint"  # FMNIST
    #     weight_decay = 0  # FMNIST
    #     prior_precision = 1 / prior_var


### PS-MAP
if method == "psmap":
    print(f"\n\n### PS-MAP ###\n")
    stochastic = False

    # learning_rate = 0.005
    # momentum = 0.9
    # alpha = 0.05  # CIFAR10-224 pretrained model
    # # alpha = 0.1  # CIFAR10
    # # alpha = 0.1  # FMNIST
    
    # reg_type = "parameter_norm"

    # weight_decay = 0
    # context_points = None
    # prior_mean = 0 if "Pretrained" not in model_name else "Pretrained Mean"
    # prior_precision = 0.00068
    # prior_var = 1 / prior_precision
    # reg_scale = 1


### FSVI
if method == "fsvi":
    print(f"\n\n### FSVI ###\n")
    stochastic = True

    # learning_rate = 0.005
    # momentum = 0.9
    # alpha = 0.1

    # reg_type = "function_prior"

    # reg_scale = 1
    # prior_mean = 0 if "Pretrained" not in model_name else "Pretrained Mean"
    # prior_var = 500000  # CIFAR10
    # # prior_var = 100000  # FMNIST
    # prior_precision = 1 / prior_var
    # context_points = "joint"
    # weight_decay = 0    


### PSVI
if method == "psvi":
    print(f"\n\n### PSVI ###\n")
    stochastic = True

    # learning_rate = 0.005
    # momentum = 0.9
    # alpha = 0.1

    # reg_type = "parameter_kl"

    # reg_scale = 1
    # prior_mean = 0 if "Pretrained" not in model_name else "Pretrained Mean"
    # prior_var = 1
    # prior_precision = 1 / prior_var
    # weight_decay = 0
    # context_points = None


print(f"\n\nDataset: {dataset}\n")
print(f"Architecture: {model_name}")
print(f"act_fn: {act_fn}")
print(f"block_class: {block_class}")
print(f"num_blocks: {num_blocks}")
print(f"c_hidden: {c_hidden}\n")

print(f"reg_type: {reg_type}")
print(f"stochastic: {stochastic}")
print(f"num_epochs: {num_epochs}")
print(f"learning_rate: {learning_rate}")
print(f"momentum: {momentum}")
print(f"alpha: {alpha}")
print(f"batch_size: {batch_size}")
print(f"context_set_size: {context_set_size}")
print(f"mc_samples_llk: {mc_samples_llk}")
print(f"reg_scale: {reg_scale}")
print(f"prior_mean: {prior_mean}")
print(f"prior_var: {prior_var}")
print(f"prior_precision: {prior_precision}")
print(f"weight_decay: {weight_decay}")
print(f"linearize: {linearize}")
print(f"context_points: {context_points}\n")


resnet_trainer, resnet_results = train_classifier(
    model_name=model_name,
    model_class=model_class,
    model_hparams={
                        "num_classes": num_classes,
                        "c_hidden": c_hidden,
                        "num_blocks": num_blocks,
                        "act_fn": act_fn,
                        "block_class": block_class,
                        "linearize": linearize,
                        },
    optimizer_name="SGD",
    optimizer_hparams={
                        "lr": learning_rate,
                        "momentum": momentum,
                        "alpha": alpha,
                        "weight_decay": weight_decay,
                        },
    objective_hparams={
                        "stochastic": stochastic,
                        "reg_type": reg_type,
                        "reg_scale": reg_scale,
                        "prior_mean": prior_mean,
                        "prior_var": prior_var,
                        "context_points": context_points,
                        "forward_points": forward_points,
                        "mc_samples_llk": mc_samples_llk,
                        "mc_samples_reg": mc_samples_reg,
                        "training_dataset_size": training_dataset_size,
                        "batch_size": batch_size,
                        "init_logvar": init_logvar,
                        "init_final_layer_bias_logvar": init_final_layer_bias_logvar,
                        "prior_feature_logvar": prior_feature_logvar,
                        "pretrained_prior": pretrained_prior,
                        },
    other_hparams={
                    "evaluate": evaluate,
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "context_set_size": context_set_size,
                    "num_epochs": num_epochs,
                    "seed": seed,
                    "mc_samples_eval": mc_samples_eval,
                    },
    exmp_inputs=jax.device_put(
        next(iter(train_loader))[0]),
    rng_key=rng_key,
    )


print(f"\nDone\n")
