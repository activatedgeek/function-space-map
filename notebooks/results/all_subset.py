import pandas as pd
from tqdm.auto import tqdm
import wandb
import pandas as pd


def get_summary_metrics(sweep_id, filter_func=None):
    api = wandb.Api(timeout=60)
    sweep = api.sweep(sweep_id)

    data = []
    for run in tqdm(sweep.runs, desc="Runs", leave=False):
        if callable(filter_func) and not filter_func(run):
            continue
        data.append(dict(run_id=run.id, **run.config, **run.summary))

    return sweep, pd.DataFrame(data)


_m1 = get_summary_metrics("deeplearn/fspace-inference/dovhf1rz")[1]
_m1["dataset"] = "CIFAR-10"
_m1["method"] = "PS-MAP"

_m2 = get_summary_metrics("deeplearn/fspace-inference/giz46j4b")[1]
_m2["dataset"] = "FashionMNIST"
_m2["method"] = "PS-MAP"

cols = [
    "run_id",
    "dataset",
    "method",
    "train_subset",
    "seed",
    "s/test/acc",
    "s/test/sel_acc",
    "s/test/avg_nll",
    "s/test/ece",
    "s/ood_test/auc",
]
__mpsmap = (
    pd.concat([_m1, _m2], ignore_index=True)[cols]
    .reset_index()
    .rename(
        columns={
            "s/test/acc": "acc",
            "s/test/sel_acc": "sel_acc",
            "s/test/avg_nll": "nll",
            "s/test/ece": "ece",
            "s/ood_test/auc": "ood_auroc",
        }
    )
)

__mpsmap["acc"] *= 100.0
__mpsmap["ece"] *= 100.0
__mpsmap["ood_auroc"] *= 100.0

del _m1, _m2

# __mpsmap.groupby(['dataset', 'train_subset', 'method']).mean(numeric_only=True)

_m3 = get_summary_metrics("deeplearn/fspace-inference/rmt0f50a")[1]
_m3["dataset"] = "CIFAR-10"
_m3["method"] = "FSGC"

_m4 = get_summary_metrics("deeplearn/fspace-inference/sbijpte2")[1]
_m4["dataset"] = "FashionMNIST"
_m4["method"] = "FSGC"

cols = [
    "run_id",
    "dataset",
    "method",
    "train_subset",
    "seed",
    "nll_test",
    "acc_sel_test",
    "acc_test",
    "ece_test",
    "ood_auroc_entropy",
]
__mfsgc = (
    pd.concat([_m3, _m4], ignore_index=True)[cols]
    .reset_index()
    .rename(
        columns={
            "nll_test": "nll",
            "acc_sel_test": "sel_acc",
            "acc_test": "acc",
            "ece_test": "ece",
            "ood_auroc_entropy": "ood_auroc",
        }
    )
)

del _m3, _m4

# __mfsgc.groupby(['dataset', 'train_subset', 'method']).mean(numeric_only=True)

metrics = (
    pd.concat([__mpsmap, __mfsgc], ignore_index=True)
    .reset_index()
    .drop(columns=["level_0", "index"])
)
# metrics.to_csv('results/all_subset.csv', index=False)

mu = (
    metrics.groupby(["dataset", "train_subset", "method"])
    .mean(numeric_only=True)
    .drop(columns=["seed"])
)
mu.acc = mu.acc.round(1)
mu.sel_acc = mu.sel_acc.round(1)
mu.nll = mu.nll.round(2)
mu.ece = mu.ece.round(1)
mu.ood_auroc = mu.ood_auroc.round(1)

sigma = (
    metrics.groupby(["dataset", "train_subset", "method"])
    .std(numeric_only=True)
    .drop(columns=["seed"])
)
sigma.acc = sigma.acc.round(1)
sigma.sel_acc = sigma.sel_acc.round(1)
sigma.nll = sigma.nll.round(2)
sigma.ece = sigma.ece.round(1)
sigma.ood_auroc = sigma.ood_auroc.round(1)

(
    mu.astype(str) + r" $\pm$ " + sigma.astype(str)
)  # .reset_index().to_markdown('tmp.md', index=False)
