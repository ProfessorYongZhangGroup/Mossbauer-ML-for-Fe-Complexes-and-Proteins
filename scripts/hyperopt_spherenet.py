from SphereNet.mossbauer_dataset import Mossbauer

dataset = Mossbauer(root='SphereNet/dataset/')

import pandas as pd
import numpy as np
from pathlib import Path

# import ray
# from ray import tune
# from ray.train import CheckpointConfig, RunConfig, ScalingConfig
# from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment,
#                                  RayTrainReportCallback, prepare_trainer)
# from ray.train.torch import TorchTrainer
# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.schedulers import FIFOScheduler

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, STATUS_OK

import torch
torch.set_float32_matmul_precision('medium')
from dig.threedgraph.method import SphereNet, run
from dig.threedgraph.evaluation import threedEvaluator

from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score

import gc

target = 'IS'
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=189, valid_size=47, seed=0)
train_dset, val_dset = dataset[split_idx['train']], dataset[split_idx['valid']]

hyopt_dir = Path.cwd() / "hyopt"
hyopt_dir.mkdir(exist_ok=True)

def train_model(config, train_dset, val_dset):
    num_layers = int(config['num_layers'])
    hidden_channels = int(config["hidden_channels"])
    int_emb_size = int(config["int_emb_size"])
    basis_emb_size_dist = int(config["basis_emb_size_dist"])
    basis_emb_size_angle = int(config["basis_emb_size_angle"])
    basis_emb_size_torsion = int(config["basis_emb_size_torsion"])
    out_emb_channels = int(config["out_emb_channels"])
    num_spherical = int(config["num_spherical"])
    num_radial = int(config["num_radial"])
    envelope_exponent = int(config["envelope_exponent"])
    num_output_layers = int(config["num_output_layers"])
    cutoff = float(config["cutoff"])
    lr = config["lr"]
    lr_decay = config["lr_decay"]
    lr_decay_step = int(config["lr_decay_step"])

    print(config)
    
    model = SphereNet(energy_and_force=False, cutoff=cutoff, num_layers=num_layers,
                      hidden_channels=hidden_channels, out_channels=1, int_emb_size=int_emb_size,
                      basis_emb_size_dist=basis_emb_size_dist, basis_emb_size_angle=basis_emb_size_angle, 
                      basis_emb_size_torsion=basis_emb_size_torsion, out_emb_channels=out_emb_channels, 
                      num_spherical=num_spherical, num_radial=num_radial, envelope_exponent=envelope_exponent,
                      num_before_skip=1, num_after_skip=2, num_output_layers=num_output_layers, use_extra_node_feature=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_func = torch.nn.L1Loss()
    evaluation = threedEvaluator()
    
    run3d = run()
    run3d.run(device, train_dset, val_dset, val_dset, model, loss_func, evaluation,
              epochs=30, batch_size=32, vt_batch_size=32, lr=lr, lr_decay_factor=lr_decay, lr_decay_step_size=lr_decay_step)
    
    valid_loader = DataLoader(val_dset, 47, shuffle=False)
    out = None
    batch = None
    for batch in valid_loader:
        batch.to(device)
        out = model(batch)
        
    mse = ((batch.IS - out[:, 0])**2).sum()/47
    mae = (abs(batch.IS - out[:, 0])).sum()/47
    r2 = r2_score(batch.IS.cpu().detach().numpy(), out[:, 0].cpu().detach().numpy())

    del run3d
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"mse": mse, "loss": mae, "r2": r2, "status": STATUS_OK}
    

# search_space = {
#     "num_layers": tune.qrandint(lower=2, upper=5, q=1),
#     "hidden_channels": tune.choice([32, 64, 128, 256]),
#     "int_emb_size": tune.choice([32, 64, 128, 256]),
#     "basis_emb_size_dist": tune.qrandint(lower=2, upper=10, q=1),
#     "basis_emb_size_angle": tune.qrandint(lower=2, upper=10, q=1),
#     "basis_emb_size_torsion": tune.qrandint(lower=2, upper=10, q=1),
#     "out_emb_channels": tune.choice([32, 64, 128, 256]),
#     "num_spherical": tune.qrandint(lower=2, upper=9, q=1),
#     "num_radial": tune.qrandint(lower=2, upper=8, q=1),
#     "envelope_exponent": tune.qrandint(lower=2, upper=7, q=1),
#     "num_output_layers": tune.qrandint(lower=2, upper=4, q=1),
#     "cutoff": tune.qrandint(lower=3, upper=10, q=1),
#     "batch_size": tune.choice([8, 16, 32, 64]),
#     "lr": tune.loguniform(1e-5, 1e-3),
#     "lr_decay": tune.uniform(0.5, 0.95),
#     "lr_decay_step": tune.qrandint(lower=10, upper=100, q=10),
# }

search_space = hp.choice('a', [
    {
        "num_layers": hp.quniform('a1', 2, 4, 1),
        "hidden_channels": hp.choice('a2', [32, 64, 128]),
        "int_emb_size": hp.choice('a3', [32, 64, 128]),
        "basis_emb_size_dist": hp.quniform('a4', 2, 10, 1),
        "basis_emb_size_angle": hp.quniform('a5', 2, 10, 1),
        "basis_emb_size_torsion": hp.quniform('a6', 2, 10, 1),
        "out_emb_channels": hp.choice('a7', [32, 64, 128]),
        "num_spherical": hp.quniform('a8', 2, 9, 1),
        "num_radial": hp.quniform('a9', 2, 8, 1),
        "envelope_exponent": hp.quniform('a10', 2, 7, 1),
        "num_output_layers": hp.quniform('a11', 2, 3, 1),
        "cutoff": hp.quniform('a12', 3, 10, 1),
        "lr": hp.uniform('a14', 1e-5, 1e-3),
        "lr_decay": hp.uniform('a15', 0.5, 0.95),
        "lr_decay_step": hp.quniform('a16', 10, 100, 10),
    }
])

best=fmin(lambda args: train_model(args, train_dset, val_dset), search_space, algo=tpe.suggest, max_evals=5)

print(best)
print(space_eval(search_space, best))

# ray.init()

# scheduler = FIFOScheduler()

# # scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
# # checkpoint_config = CheckpointConfig(num_to_keep=5,
# #                                      checkpoint_score_attribute="val_loss",
# #                                      checkpoint_score_order="min",
# #                                     )
# run_config = RunConfig(storage_path=hyopt_dir / "ray_results")

# search_alg = HyperOptSearch(
#     n_initial_points=1, # number of random evaluations before tree parzen estimators
#     random_state_seed=42,
# )

# tune_config = tune.TuneConfig(
#     metric="mae",
#     mode="min",
#     num_samples=1, # number of trials to run
#     scheduler=scheduler,
#     search_alg=search_alg,
#     trial_dirname_creator=lambda trial: str(trial.trial_id), # shorten filepaths
# )

# tuner = tune.Tuner(
#     lambda config: train_model(config, train_dset, val_dset),
#     param_space=search_space,
#     tune_config=tune_config,
#     run_config=run_config
# )

# results = tuner.fit()