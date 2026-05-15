# Machine Learning Predictions of Mössbauer Parameters of Fe-containing Molecular Complexes and Protein Models with Close-to-DFT Accuracy

This repository provides the iron complex structures collected and the code ran to produce the results for our recently published research. Additionally, 
trained models for Chemprop and SphereNet are included, as the best Ridge regression and Support Vector regression models are easily replicable with the 
given code. 

## Installation

The following libraries must be installed in order to run included Jupyter notebooks and scripts:

> [!WARNING]
> Certain minor changes to the code of SphereNet were made in order to accommodate our prediction task. Please install the files `spherenet_files/run.py` at `[env]/lib/python3.12/site-packages/dig/threedgraph/method`
> and `spherenet_files/features.py` at `[env]/lib/python3.12/site-packages/dig/threedgraph/method/spherenet` in order to use our SphereNet model or train your own, where `[env]` is the root of the installed python environmeent.

## Citation

Find out paper at [url]. Please cite our paper if you decide to use our code, dataset, or models for your own work.
