#!/usr/bin/env python
import argparse
import pathlib
import sys
import os
import os.path as osp
import shutil
import subprocess
import logging

def optimize(input_file: pathlib.Path, charge, spin, temp, output_folder):
    prev_dir = os.getcwd()
    os.chdir(output_folder)

    param = \
f"""$chrg {int(charge)}
$spin {int(2*spin)}
$gfn
   method=1
   scc=true
   periodic=false
   dispscale=1.0000000000000000
$scc
   maxiterations=2000
   temp=100
   broydamp=0.40000000000000002
$opt
   optlevel=normal
   microcycle=20
   maxcycle=0
   maxdispl=1.0000000000000000
   hlow=0.10000000000000000E-1
   hessian=old
   s6=20.000000000000000
   kstretch=0.40000000000000002
   kbend   =0.13000000000000000
   ktorsion=0.74999999999999997E-2
   koutofp =0.0000000000000000
   kvdw    =0.0000000000000000
   kes     =0.0000000000000000
   rcut    =8.3666002653407556
   ts=0
   tsroot=0
   exact rf=false
   average conv=false
$thermo
   temp={temp}
   sthr=50.000000000000000
   imagthr=-20.000000000000000
   scale=1.0000000000000000
$md
   temp=298.14999999999998
   time=50.000000000000000
   dump=50.000000000000000
   velo=0
   nvt=1
   skip=500
   step=4.0000000000000000
   hmass=4
   shake=2
   sccacc=2.0000000000000000
   forcewrrestart=0
$siman
   dump=1000.0000000000000
   n=3
   ewin=20.000000000000000
   temp=1000.0000000000000
   enan=0
   check=0
$hess
   sccacc=0.29999999999999999
   step=0.50000000000000001E-2
   scale=1.0000000000000000
$modef
   n=31
   step=1.0000000000000000
   updat=0.20000000000000001
   local=0
   vthr=0.0000000000000000
   prj=0
   mode=7
$cube
   step=0.40000000000000002
   pthr=0.50000000000000003E-1
$symmetry
   desy=0.10000000000000001
   maxat=200
$embedding
   at=7
   es=true
$write
   esp=false
   mos=false
   gbw=false
   tm mos=false
   tm basis=false
   lmo=false
   density=false
   spin population=true
   spin density=false
   fod=false
   fod population=false
   wiberg=true
   wbo fragments=false
   dipole=false
   charges=false
   mulliken=false
   orbital energies=false
   inertia=false
   distances=false
   angles=false
   torsions=false
   final struct=true
   geosum=false
   stm=false
   modef=false
   gbsa=false
   vib_normal_modes=false
   hessian.out=false
$external
$path
   nrun=3
   npoint=50
   anopt=3
   kpush=0.50000000000000003E-1
   kpull=-0.40000000000000001E-1
   ppull=0.50000000000000003E-1
   alp=0.69999999999999996
$split
   fragment: 1,1-78
$wall
   potential=polynomial
   alpha=30
   beta=6.0000000000000000
   temp=300.00000000000000
   autoscale=1.0000000000000000
   axisshift=3.5000000000000000
$reactor
   max=50
   density=1.0000000000000000 # in kg/L
   kpush=0.50000000000000003E-1
   alp=0.69999999999999996
$end
"""
    
    if(not osp.isfile('param.inp')):
        with open('param.inp', 'w') as file:
            file.write(param)

    if(not osp.isfile('xtbopt.mol')):
        output = subprocess.run(["xtb", 
                                 str(input_file), 
                                 "--cosmo", "32.7", 
                                 "--opt", 
                                 "--input", "param.inp"], 
                                encoding='utf-8', 
                                stdout=subprocess.PIPE) 
    
        with open('xtb.out', 'w') as file:
            file.write(output.stdout)

    if(not osp.isfile('tblite-data.npz')):
        output = subprocess.run(["tblite", 
                                 "run", 
                                 "--cpcm", "32.7", 
                                 "--spin-polarized", 
                                 "--post-processing", "xtbml_xyz",
                                 "--charge", f"{int(charge)}",
                                 "--spin", f"{int(2*spin)}",
                                 "xtbopt.mol"], 
                                encoding='utf-8', 
                                stdout=subprocess.PIPE)
    
        print(output.stdout)
    
        with open('tblite.out', 'w') as file:
            file.write(output.stdout)

    os.chdir(prev_dir)
    
def load_mol(charge: int, spin: int, fe_spin: int, temp, output_folder):
    if(osp.exists((file_path := osp.join(output_folder, 'xtbtopo.mol')))):
        pass
    else:
        file_path = osp.join(output_folder, 'xtbopt.mol')

    mol = chem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
    mol.UpdatePropertyCache(strict=False)

    chem.SanitizeMol(mol, c.SanitizeFlags.SANITIZE_FINDRADICALS | \
                          c.SanitizeFlags.SANITIZE_SETAROMATICITY | \
                          c.SanitizeFlags.SANITIZE_SETCONJUGATION | \
                          c.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | \
                          c.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)

    desc_path = osp.join(output_folder, 'tblite-data.npz')
    post_processing = np.load(desc_path)

    charges = post_processing['tblite0_q_A_alpha'] + post_processing['tblite0_q_A_beta']

    d_density_alpha = post_processing['tblite0_p_d_alpha']
    p_density_alpha = post_processing['tblite0_p_p_alpha']
    s_density_alpha = post_processing['tblite0_p_s_alpha']

    d_density_beta = post_processing['tblite0_p_d_beta']
    p_density_beta = post_processing['tblite0_p_p_beta']
    s_density_beta = post_processing['tblite0_p_s_beta']

    chem_pot_alpha = post_processing['tblite0_chem_pot_alpha']
    chem_pot_beta = post_processing['tblite0_chem_pot_beta']
    
    gap_alpha = post_processing['tblite0_gap_alpha']
    gap_beta = post_processing['tblite0_gap_beta']

    hoao_a = post_processing['tblite0_HOAO_alpha']
    hoao_b = post_processing['tblite0_HOAO_beta']

    luao_a = post_processing['tblite0_LUAO_alpha']
    luao_b = post_processing['tblite0_LUAO_beta']

    V_f_final = np.column_stack((charges,
                                 d_density_alpha, p_density_alpha, s_density_alpha, 
                                 d_density_beta, p_density_beta, s_density_beta,
                                 chem_pot_alpha, chem_pot_beta, 
                                 gap_alpha, gap_beta,
                                 hoao_a, hoao_b, luao_a, luao_b))

    charge_one_hot = list(np.eye(8)[int(charge) + 4])
    spin_one_hot = list(np.eye(6)[int(2*spin)])
    fe_spin_one_hot = list(np.eye(6)[int(2*fe_spin)])

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
                  post_processing[f'tblite0_ext_qm_e_alpha'][iron_index],
                  post_processing[f'tblite0_ext_qm_e_beta'][iron_index],
                  post_processing[f'tblite0_ext_qm_Z_alpha'][iron_index],
                  post_processing[f'tblite0_ext_qm_Z_beta'][iron_index]]

    threed_list = list(Descriptors3D.CalcMolDescriptors3D(mol).values())
    
    x_d_final = np.asarray(charge_one_hot + spin_one_hot + fe_spin_one_hot + threed_list + quadrupole + [temp, charges[iron_index]])
    
    return data.MoleculeDatapoint(mol, 0, x_d=x_d_final, V_f=V_f_final)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='predictor',
                    description='Predict Mossbauer spectroscopy parameters for iron complex structures')

    parser.add_argument('input', type=pathlib.Path, 
                        help='Input file or folder of iron complex structures to predict Mossbauer parameters.')

    parser.add_argument('--charge', type=int, 
                        help='Charge of iron complex.')

    parser.add_argument('--spin', type=float, 
                        help='Spin multiplicity of iron complex.')

    parser.add_argument('--fe-spin', type=float, 
                        help='Spin multiplicity of iron atom.')
    
    parser.add_argument('--temperature', type=float, 
                        help='Temperature of sample in K.')
    
    parser.add_argument('--output', type=pathlib.Path,
                        help='Directory for output files created during prediction process.')

    parser.add_argument('--batch-file', type=pathlib.Path,
                        help='List of charge, spin, and temperature for every complex being predicted.')

    args = parser.parse_args()

    # Check existence of input and output
    if(not osp.isfile(args.input) and not osp.isdir(args.input)):
        sys.exit('Input is not a file or directory.')

    absolute_input = osp.abspath(args.input)
    
    batch_file = None
    if(osp.isdir(absolute_input)):   
        if(args.batch_file is None):
            batch_file = osp.join(absolute_input, 'cst.csv')
        else:
            if(not osp.isfile(args.batch_file)):
                sys.exit('Batch file is not a file.')

            batch_file = osp.abspath(args.batch_file)
    else:
        if(args.charge is None):
            sys.exit('Charge must be set if an individual structure is input.')
        if(args.spin is None):
            sys.exit('Spin must be set if an individual structure is input.')
        if(args.temperature is None):
            sys.exit('Temperature must be set if an individual structure is input.')
        if(args.temperature < 0 or args.temperature > 500):
            sys.exit('Temperature must be in the range 0-500 K.')

    output_folder = None
    if(args.output is not None):
        if(not osp.isdir(args.output)):
            if(osp.isfile(args.output)):
                sys.exit('Output is not a directory.')

            os.makedirs(args.output)
        output_folder = osp.abspath(args.output)
    else:
        output_folder = os.getcwd()

    if(shutil.which('xtb') is None):
        sys.exit('xtb must be installed and in path.')

    if(shutil.which('tblite') is None):
        sys.exit('tblite must be installed and in path.')

    import pandas as pd
    import numpy as np
    
    from rdkit.Chem import AllChem as chem
    from rdkit import Chem as c
    from rdkit.Chem import Descriptors3D
    
    from chemprop import data, featurizers, models, nn
    from featurizers.atom import TMCAtomFeaturizer
    from featurizers.bond import TMCBondFeaturizer
    
    import torch
    
    from lightning import pytorch as pl

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    mol_list = []
    folder_list = []
    if(batch_file is not None):
        cst = pd.read_csv(batch_file, index_col=0)
        for file in cst.index:
            print(cst.loc[file])
            charge, spin, fe_spin, temp = tuple(cst.loc[file])
            new_output_folder = osp.join(output_folder, pathlib.Path(file).stem)
            if not(os.path.exists(new_output_folder) and os.path.isdir(new_output_folder)):
                os.makedirs(new_output_folder)
            
            folder_list.append(new_output_folder)
            optimize(osp.join(absolute_input, file), charge, spin, temp, new_output_folder)
            mol_list.append(load_mol(charge, spin, fe_spin, temp, new_output_folder))
    else:
        folder_list.append(output_folder)
        optimize(absolute_input, args.charge, args.spin, args.temperature, output_folder)
        mol_list.append(load_mol(args.charge, 
                                 args.spin, 
                                 args.fe_spin if args.fe_spin is not None else args.spin,
                                 args.temperature, output_folder))

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(
                            atom_featurizer=TMCAtomFeaturizer.complex(),
                            bond_featurizer=TMCBondFeaturizer(),
                            extra_atom_fdim=mol_list[0].V_f.shape[1])

    predict_dset = data.MoleculeDataset(np.asarray(mol_list), featurizer)
    predict_loader = data.build_dataloader(predict_dset, shuffle=False, num_workers=10, batch_size=len(mol_list))

    model = models.MPNN.load_from_checkpoint('models/best.ckpt', map_location=torch.device('cpu'))

    trainer = pl.Trainer(
        enable_progress_bar=True,
        logger=False,
        accelerator="cpu",
        devices=1
    )
    
    out = trainer.predict(model, predict_loader, return_predictions=True)

    for i, f in enumerate(folder_list):
        print(f'\n{osp.basename(f)}.mol')
        output = \
f"""Isomer shift: {out[0][i, 0]} mm/s
Quadrupole splitting: {out[0][i, 1]} mm/s"""
        print(output)
        
        with open(osp.join(f, 'moss.out'), 'w') as file:
            file.write(output)
