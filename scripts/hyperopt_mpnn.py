from rdkit.Chem import AllChem as chem
from rdkit import Chem as c
from rdkit.Chem import Descriptors3D
import os.path as osp
import pandas as pd 

data_folder = 'SphereNet/dataset/mossbauer/raw_gfn1'
smiles_csv = pd.read_csv(osp.join(data_folder, 'annotations.tsv'),sep='\t')

file_list = list(smiles_csv.iloc[:,0])
mol_list = []

for file in file_list:
    file_path = None
    if(osp.exists((file_path := osp.join(data_folder, file[:-4], 'xtbtopo.mol')))):
        pass
    else:
        file_path = osp.join(data_folder, file[:-4], 'xtbopt.mol')

    mol = chem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
    # Regenerate computed properties like implicit valence and ring information
    mol.UpdatePropertyCache(strict=False)

    # Apply several sanitization rules
    chem.SanitizeMol(mol,c.SanitizeFlags.SANITIZE_FINDRADICALS| \
                         c.SanitizeFlags.SANITIZE_SETAROMATICITY| \
                         c.SanitizeFlags.SANITIZE_SETCONJUGATION| \
                         c.SanitizeFlags.SANITIZE_SETHYBRIDIZATION| \
                         c.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
    mol_list.append(mol)

import pandas as pd
import numpy as np
from pathlib import Path

from lightning import pytorch as pl

import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment,
                                 RayTrainReportCallback, prepare_trainer)
from ray.train.torch import TorchTrainer
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import FIFOScheduler

from chemprop import data, featurizers, models, nn

num_workers = 15 # number of workers for dataloader. 0 means using main process for data loading
target_columns = ['IS','QS'] # classification of activity (either 0 or 1)
feature_columns = ['CHARGE', 'SPIN', 'FE SPIN', 'TEMP'] # name of feature columns in csv

ys = smiles_csv.loc[:, target_columns].values
x_ds = smiles_csv.loc[:, feature_columns].values
x_dsize = 0

datapoint_list = []
for mol, file, y, x_d in zip(mol_list, file_list, ys, x_ds):
    desc_path = osp.join(data_folder, file[:-4], 'tblite-data.npz')
    post_processing = np.load(desc_path)

    charges = post_processing['tblite0_q_A_alpha'] + post_processing['tblite0_q_A_beta']

    d_density_alpha = post_processing['tblite0_p_d_alpha']
    p_density_alpha = post_processing['tblite0_p_p_alpha']
    s_density_alpha = post_processing['tblite0_p_s_alpha']

    d_density_beta = post_processing['tblite0_p_d_beta']
    p_density_beta = post_processing['tblite0_p_p_beta']
    s_density_beta = post_processing['tblite0_p_s_beta']

    spin_density = d_density_alpha + p_density_alpha + s_density_alpha - (d_density_beta + p_density_beta + s_density_beta)

    chem_pot_alpha = post_processing['tblite0_chem_pot_alpha']
    chem_pot_beta = post_processing['tblite0_chem_pot_beta']
    
    gap_alpha = post_processing['tblite0_gap_alpha']
    gap_beta = post_processing['tblite0_gap_beta']

    hoao_a = post_processing['tblite0_HOAO_alpha']
    hoao_b = post_processing['tblite0_HOAO_beta']

    luao_a = post_processing['tblite0_LUAO_alpha']
    luao_b = post_processing['tblite0_LUAO_beta']

    V_f_final = np.column_stack((charges, spin_density,
                                 d_density_alpha, p_density_alpha, s_density_alpha, 
                                 d_density_beta, p_density_beta, s_density_beta,
                                 chem_pot_alpha, chem_pot_beta, 
                                 gap_alpha, gap_beta,
                                 hoao_a, hoao_b, luao_a, luao_b))
    
    charge_one_hot = list(np.eye(8)[int(x_d[0] + 4)])
    spin_one_hot = list(np.eye(6)[int(x_d[1] * 2)])
    fe_spin_one_hot = list(np.eye(6)[int(x_d[2] * 2)])

    iron_index = 0
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomicNum() == 26:
            iron_index = i
            break
           
    quadrupole = [post_processing[f'tblite0_qm_A_{x+13*' '}_alpha'][iron_index] + 
                  post_processing[f'tblite0_qm_A_{x+13*' '}_beta'][iron_index] 
                  for x in ['zz','yy','xx','xy','xz','yz']] + \
                 [post_processing[f'tblite0_ext_qm_e_{x+9*' '}_alpha'][iron_index] + 
                  post_processing[f'tblite0_ext_qm_e_{x+9*' '}_beta'][iron_index] 
                  for x in ['zz','yy','xx','xy','xz','yz']] + \
                 [post_processing[f'tblite0_ext_qm_Z_{x+9*' '}_alpha'][iron_index] + 
                  post_processing[f'tblite0_ext_qm_Z_{x+9*' '}_beta'][iron_index] 
                  for x in ['zz','yy','xx','xy','xz','yz']] + \
                 [post_processing[f'tblite0_qm_A_alpha'][iron_index], 
                  post_processing[f'tblite0_qm_A_beta'][iron_index],
                  post_processing[f'tblite0_ext_qm_A_alpha'][iron_index],
                  post_processing[f'tblite0_ext_qm_A_beta'][iron_index],
                  post_processing[f'tblite0_ext_qm_e_alpha'][iron_index],
                  post_processing[f'tblite0_ext_qm_e_beta'][iron_index],
                  post_processing[f'tblite0_ext_qm_Z_alpha'][iron_index],
                  post_processing[f'tblite0_ext_qm_Z_beta'][iron_index]]

    threed_list = list(Descriptors3D.CalcMolDescriptors3D(mol).values())
    
    x_d_final = np.asarray(charge_one_hot + spin_one_hot + fe_spin_one_hot + threed_list + quadrupole + [x_d[3], charges[iron_index]])    

    x_dsize = x_d_final.shape[0]
    datapoint_list.append(data.MoleculeDatapoint(mol, y, x_d=x_d_final, V_f=V_f_final))

all_data = np.asarray(datapoint_list)
hyopt_dir = Path.cwd() / "hyopt"
hyopt_dir.mkdir(exist_ok=True)

mols = [d.mol for d in all_data]
train_indices, val_indices, test_indices = data.make_split_indices(mols, 'random', (0.8, 0.2, 0.0))
train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

import torch
torch.set_float32_matmul_precision('high')

from chemprop.nn.transforms import ScaleTransform, GraphTransform, UnscaleTransform

from featurizers.atom import TMCAtomFeaturizer
from featurizers.bond import TMCBondFeaturizer

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(
                                atom_featurizer=TMCAtomFeaturizer.complex(),
                                bond_featurizer=TMCBondFeaturizer(),
                                extra_atom_fdim=V_f_final.shape[1])

train_dset = data.MoleculeDataset(train_data[0], featurizer)
targets_scaler = train_dset.normalize_targets()
X_d_scaler = train_dset.normalize_inputs("X_d")
V_f_scaler = train_dset.normalize_inputs("V_f")

val_dset = data.MoleculeDataset(val_data[0], featurizer)
val_dset.normalize_targets(targets_scaler)
val_dset.normalize_inputs("X_d", X_d_scaler)
val_dset.normalize_inputs("V_f", V_f_scaler)

def train_model(config, train_dset, val_dset, num_workers, targets_scaler, X_d_scaler, V_f_scaler):
    depth = int(config["depth"])
    ffn_hidden_dim = int(config["ffn_hidden_dim"])
    ffn_num_layers = int(config["ffn_num_layers"])
    message_hidden_dim = int(config["message_hidden_dim"])
    
    train_dset.cache = True

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, shuffle=True)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)

    n_V_features = featurizer.atom_fdim - featurizer.extra_atom_fdim
    V_f_transform = ScaleTransform.from_standard_scaler(V_f_scaler, pad=n_V_features)
    graph_transform = GraphTransform(V_f_transform, ScaleTransform([0],[1]))
    
    X_d_transform = ScaleTransform.from_standard_scaler(X_d_scaler)

    output_transform = UnscaleTransform.from_standard_scaler(targets_scaler)

    mp = nn.BondMessagePassing(d_h=message_hidden_dim, depth=depth,
                               d_v=featurizer.atom_fdim, 
                               d_e=featurizer.bond_fdim, 
                               graph_transform=graph_transform,
                               dropout=config["dropout_mpnn"])
    agg = config["aggregation"](output_size=message_hidden_dim)
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim + x_dsize,
        n_tasks=len(target_columns),
        output_transform=output_transform,
        dropout=config["dropout_ffn"],
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers)
    batch_norm = True
    metric_list = [nn.metrics.MAE(), nn.metrics.MSE(), nn.metrics.R2Score()]
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=X_d_transform, init_lr=config["lr"])

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=False,
        accelerator="gpu",
        devices=1,
        max_epochs=200, # number of epochs to train for
        strategy=RayDDPStrategy(find_unused_parameters=True, static_graph=True),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(mpnn, train_loader, val_loader)

search_space = {
    "depth": tune.qrandint(lower=1, upper=5, q=1),
    "ffn_hidden_dim": tune.qrandint(lower=200, upper=1000, q=200),
    "ffn_num_layers": tune.qrandint(lower=1, upper=5, q=1),
    "message_hidden_dim": tune.qrandint(lower=200, upper=1000, q=200),
    "aggregation": tune.choice([nn.MeanAggregation, nn.NormAggregation, nn.AttentiveAggregation]),
    "lr": tune.loguniform(1e-5, 1e-3),
    "dropout_ffn": tune.quniform(0.0, 0.5, 0.1),
    "dropout_mpnn": tune.quniform(0.0, 0.5, 0.1),
}

ray.init()

scheduler = FIFOScheduler()

scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
checkpoint_config = CheckpointConfig(num_to_keep=5,
                                     checkpoint_score_attribute="val_loss",
                                     checkpoint_score_order="min",
                                    )
run_config = RunConfig(checkpoint_config=checkpoint_config,
                       storage_path=hyopt_dir / "ray_results",
                      )

ray_trainer = TorchTrainer(
    lambda config: train_model(
        config, train_dset, val_dset, num_workers, targets_scaler, X_d_scaler, V_f_scaler
    ),
    scaling_config=scaling_config,
    run_config=run_config,
)

search_alg = HyperOptSearch(
    n_initial_points=1, # number of random evaluations before tree parzen estimators
    random_state_seed=42,
)

# OptunaSearch is another search algorithm that can be used
# search_alg = OptunaSearch()

tune_config = tune.TuneConfig(
    metric="val_loss",
    mode="min",
    num_samples=400, # number of trials to run
    scheduler=scheduler,
    search_alg=search_alg,
    trial_dirname_creator=lambda trial: str(trial.trial_id), # shorten filepaths

)

tuner = tune.Tuner(
    ray_trainer,
    param_space={
        "train_loop_config": search_space,
    },
    tune_config=tune_config,
)

results = tuner.fit()
