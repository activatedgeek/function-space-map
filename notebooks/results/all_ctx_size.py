import pandas as pd
from tqdm.auto import tqdm
import wandb
import pandas as pd

def get_summary_metrics(sweep_id, filter_func=None):
  api = wandb.Api(timeout=60)
  sweep = api.sweep(sweep_id)

  data = []
  for run in tqdm(sweep.runs, desc='Runs', leave=False):
    if callable(filter_func) and not filter_func(run):
      continue
    data.append(dict(run_id=run.id, **run.config, **run.summary))

  return sweep, pd.DataFrame(data)

cols = ['run_id', 'seed', 'context_set_size', 'context_subset',
        'nll_test', 'acc_sel_test', 'acc_test', 'ece_test', 'ood_auroc_entropy']

_m1 = get_summary_metrics('deeplearn/fspace-inference/kmzrb0cz')[1][cols]
_m1['dataset'] = 'CIFAR-10'

_m2 = get_summary_metrics('deeplearn/fspace-inference/9wisuedm')[1][cols]
_m2['dataset'] = 'FashionMNIST'

metrics = pd.concat([_m1,_m2], ignore_index=True).rename(
    columns={ 'nll_test': 'nll', 'acc_sel_test': 'sel_acc', 'acc_test': 'acc', 'ece_test': 'ece', 'ood_auroc_entropy': 'ood_auroc'})
# metrics.to_csv('results/all_ctx_size.csv', index=False)

cols = ['acc', 'sel_acc', 'nll', 'ece', 'ood_auroc']

mu = metrics.groupby(['dataset', 'context_subset', 'context_set_size']).mean(numeric_only=True).drop(columns=['seed'])[cols]
mu.acc = mu.acc.round(1)
mu.sel_acc = mu.sel_acc.round(1)
mu.nll = mu.nll.round(2)
mu.ece = mu.ece.round(1)
mu.ood_auroc = mu.ood_auroc.round(1)

sigma = metrics.groupby(['dataset', 'context_subset', 'context_set_size']).std(numeric_only=True).drop(columns=['seed'])[cols]
sigma.acc = sigma.acc.round(1)
sigma.sel_acc = sigma.sel_acc.round(1)
sigma.nll = sigma.nll.round(2)
sigma.ece = sigma.ece.round(1)
sigma.ood_auroc = sigma.ood_auroc.round(1)