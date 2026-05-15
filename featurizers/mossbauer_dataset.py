import os
import os.path as osp
from io import StringIO
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data

def get_atomic_number(symbol):
        atom_map = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
            "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
            "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
            "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
            "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
            "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59,
            "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67,
            "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
            "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
            "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
            "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
            "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106,
            "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113,
            "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
        }
        return atom_map.get(symbol)

class Mossbauer(InMemoryDataset):
    def __init__(self, root='dataset/', transform=None, pre_transform=None, pre_filter=None):
        self.folder = os.path.join(root, 'mossbauer')
        
        super(Mossbauer, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
    @property
    def raw_files_names(self):
        return 'annotations.tsv'

    @property
    def processed_file_names(self):
        return 'mossbauer_pyg.pt'

    def process(self):
        annotations = pd.read_csv(osp.join(self.raw_dir, self.raw_files_names), sep='\t')

        data_list = []
        for i, row in annotations.iterrows():
            print(row.iloc[0])
            folder_path = osp.join(self.raw_dir, str(row.iloc[0][:-4]))
            raw_xyz = []
            with open(osp.join(folder_path, 'xtbopt.mol'), 'r') as f:
                while (line := f.readline()):
                    raw_xyz.append(line[:33])

            xyz = pd.read_csv(StringIO('\n'.join(raw_xyz[4:int(raw_xyz[3][:3])+4])), sep='\\s+', names=['x','y','z','atom'])
            z = torch.tensor(xyz.loc[:, 'atom'].apply(get_atomic_number).values, dtype=torch.int64)
            pos = torch.tensor(xyz.loc[:, 'x':'z'].values,dtype=torch.float32)
            y = [torch.tensor(row.iloc[target],dtype=torch.float32) for target in [1, 2]]

            desc_path = osp.join(folder_path, 'tblite-data.npz')
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

            iron_index = xyz.loc[:, 'atom'].apply(get_atomic_number)
            iron_index = iron_index[iron_index == 26].index[0]

            quadrupole = [post_processing[f'tblite0_qm_A_{x+13*' '}_alpha'][iron_index] + 
                  post_processing[f'tblite0_qm_A_{x+13*' '}_beta'][iron_index] 
                  for x in ['zz','yy','xx','xy','xz','yz']] + \
                 [post_processing[f'tblite0_qm_A_alpha'][iron_index], 
                  post_processing[f'tblite0_qm_A_beta'][iron_index]]

            temp = row.iloc[3]
            charge_one_hot = list(np.eye(8)[int(row.iloc[4] + 4)])
            spin_one_hot = list(np.eye(6)[int(row.iloc[5] * 2)])
            fe_spin_one_hot = list(np.eye(6)[int(row.iloc[6] * 2)])
            
            x_d = np.asarray(([charge_one_hot + spin_one_hot + fe_spin_one_hot + quadrupole + [temp, charges[iron_index]]]) * charges.shape[0])

            node_features = torch.as_tensor(np.column_stack(tuple([charges, 
                                                             d_density_alpha, p_density_alpha, s_density_alpha, 
                                                             d_density_beta, p_density_beta, s_density_beta,
                                                             chem_pot_alpha, chem_pot_beta, 
                                                             gap_alpha, gap_beta,
                                                             hoao_a, hoao_b, luao_a, luao_b, x_d])), dtype=torch.float32)
            
            data = Data(pos=pos, z=z, y=y[0], IS=y[0], QS=y[1], node_feature=node_features)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict
