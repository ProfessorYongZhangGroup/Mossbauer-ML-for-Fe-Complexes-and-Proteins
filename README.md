# Machine Learning Predictions of Mössbauer Parameters of Fe-containing Molecular Complexes and Protein Models with Close-to-DFT Accuracy
<img width="500" height="853" alt="image" src="https://github.com/user-attachments/assets/14388431-610c-4703-98b6-7691723aed8c" />

This repository provides the iron complex structures collected and the code ran to produce the results for our recently published research. Additionally, 
trained models for Chemprop and SphereNet are included, as the best Ridge regression and Support Vector regression models are easily replicable with the 
given code. 

## Installation

The following python (v. 3.12.9) libraries must be installed in order to run included Jupyter notebooks and scripts:

- RDKit (v. 2025.03.3)
- Scikit-learn (v. 1.5.1)
- Pandas (v. 2.2.3)
- PyTorch (v. 2.7.1+cu126)
- PyTorch Lightning (v. 2.5.2)
- Chemprop (v. 2.1.2)
- DIG (v. 0.0.4)

> [!WARNING]
> Version numbers are those that were originally used. Using a different python or library version may result in failure of the code to run (PyTorch version is less important). We recommend that the libraries are installed with conda/mamba.

> [!WARNING]
> Certain minor changes to the code of SphereNet were made in order to accommodate our prediction task. Please install the files `spherenet_files/run.py` at `[env]/lib/python3.12/site-packages/dig/threedgraph/method`
> and `spherenet_files/features.py` at `[env]/lib/python3.12/site-packages/dig/threedgraph/method/spherenet` in order to use our SphereNet model or train your own, where `[env]` is the root of the installed python environmeent.

Semiempirical calculations were performed using [xtb](https://github.com/grimme-lab/xtb) (v. 6.7.1) and [tblite](https://github.com/tblite/tblite) (v. 0.5.0). Please make these available on the `PATH` environmental variable so that relevant code can run these programs.

## Citation

Find out paper at [url]. Please cite our paper if you decide to use our code, dataset, or models for your own work.
