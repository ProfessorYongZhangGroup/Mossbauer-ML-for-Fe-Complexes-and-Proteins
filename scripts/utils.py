from rdkit.Chem import AllChem as chem
from rdkit import Chem as c
from rdkit.Chem import Descriptors3D
import os.path as osp
import pandas as pd
import numpy as np
from pathlib import Path
from chemprop import data


def load_and_create_dataset_MPNN(data_folder, target_columns, feature_columns):
    smiles_csv = pd.read_csv(osp.join(data_folder, 'annotations.tsv'),sep='\t')
    
    file_list = list(smiles_csv.iloc[:,0])
    mol_list = []
    
    for file in file_list:
        file_path = None
        if(osp.exists((file_path := osp.join(data_folder, file[:-4], 'xtbtopo.mol')))):
            pass
        else:
            file_path = osp.join(data_folder, file[:-4], 'xtbopt.mol')

        #print(file_path)
    
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


    num_workers = 15
    
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
                      post_processing[f'tblite0_ext_qm_e_alpha'][iron_index],
                      post_processing[f'tblite0_ext_qm_e_beta'][iron_index],
                      post_processing[f'tblite0_ext_qm_Z_alpha'][iron_index],
                      post_processing[f'tblite0_ext_qm_Z_beta'][iron_index]]
    
        threed_list = list(Descriptors3D.CalcMolDescriptors3D(mol).values())
    
        x_d_final = np.asarray(charge_one_hot + spin_one_hot + fe_spin_one_hot + threed_list + quadrupole + [x_d[3], charges[iron_index]])
            
        x_dsize = x_d_final.shape[0]
        datapoint_list.append(data.MoleculeDatapoint(mol, y, x_d=x_d_final, V_f=V_f_final, name=file))
    
    all_data = np.asarray(datapoint_list)

    return all_data, V_f_final.shape[1] 