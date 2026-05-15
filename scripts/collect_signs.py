#!/usr/bin/env python
import argparse
import pathlib
import sys
import os
import os.path as osp
import shutil
import subprocess
import logging

def calculate_mossbauer(input_file: pathlib.Path, charge, spin):
    raw_xyz = []
    with open(input_file, 'r') as f:
        while (line := f.readline()):
            raw_xyz.append(line[:33])

    axyz = pd.read_csv(StringIO('\n'.join(raw_xyz[4:int(raw_xyz[3][:3])+4])), sep='\\s+', names=['x','y','z','atom']).loc[:, ['atom','x','y','z']]

#     heavy = \
# f"""{heavy_atoms} 0
# 6-311G*
# ****"""
#     hydrogen = \
# f"""H 0
# 6-31G*
# ****"""

#     param = \
# f"""%mem=50GB
# %nprocshared=64
# %chk=438.chk
# # b3lyp/gen output=wfn prop=efg scf=tight

# {num}

# {charge} {(spin*2) + 1}
# {axyz}

# Fe  0
#  s  6  1.00
#  257539.       .00029
#  38636.9       .00226
#  8891.44       .01152
#  2544.01       .04566
#  844.777       .14035
#  312.527       .31420
#  s  2  1.00
#  125.593       .40878
#  53.4987       .21163
#  s  1  1.00
#  17.7151      1.0
#  s  1  1.00
#  7.37677      1.0
#  s  1  1.00
#  2.01847      1.0
#  s  1  1.00
#  .779935      1.0
#  s  1  1.00
#  .114220      1.0
#  s  1  1.00
#  .041889      1.0
#  p  3  1.00
#  1678.40       .00249
#  396.392       .02015
#  128.588       .09199
#  p  3  1.00
#  49.1158       .25991
#  20.5035       .42887
#  8.98712       .32691
#  p  1  1.00
#  3.68249      1.0
#  p  1  1.00
#  1.52175      1.0
#  p  1  1.00
#  .592684      1.0
#  p  1  1.00
#  0.2023725    1.0
#  p  1  1.00
#  0.0627645    1.0
#  d  3  1.00
#  41.4526       .02511
#  11.5403       .13626
#  3.88543       .35323
#  d  1  1.00
#  1.32380      1.0
#  d  1  1.00
#  .416680      1.0
#  d  1  1.00
#  0.11330      1.0
# ****
# {heavy}
# {hydrogen}

# {num}.wfn

# """

    # with open('param.inp', 'w') as file:
    #     file.write(param)

    # if(not osp.isfile('xtbopt.mol')):
    #     output = subprocess.run(["xtb", 
    #                              str(input_file), 
    #                              "--alpb", "chloroform", 
    #                              "--opt", 
    #                              "--input", "param.inp"], 
    #                             encoding='utf-8', 
    #                             stdout=subprocess.PIPE) 
    
    #     with open('xtb.out', 'w') as file:
    #         file.write(output.stdout)

    # if(not osp.isfile('tblite-data.npz')):
    #     output = subprocess.run(["tblite", 
    #                              "run", 
    #                              "--alpb", "chloroform", 
    #                              "--spin-polarized", 
    #                              "--guess", "ceh",
    #                              "--post-processing", "xtbml_xyz",
    #                              "--charge", f"{charge}",
    #                              "--spin", f"{spin-1}",
    #                              "xtbopt.mol"], 
    #                             encoding='utf-8', 
    #                             stdout=subprocess.PIPE)
    
    #     print(output.stdout)
    
    #     with open('tblite.out', 'w') as file:
    #         file.write(output.stdout)
    print(axyz)

if __name__ == '__main__':
    annotations = pd.read_csv('./raw/annotations.tsv', sep='\t')
    annotations.index = annotations['FILE']
    
    for root, _, files in os.walk('./raw'):
        for file in files:
            if file == 'xtbopt.mol':
                charge = annotations.loc[f"{root[6:]}.mol", 'CHARGE']
                spin = annotations.loc[f"{root[6:]}.mol", 'SPIN STATE']
                calculate_mossbauer(osp.join(root, file), charge, spin)
                break
        break
