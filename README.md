# Reproducing the PredFull model

This repo contains code to reproduce the [PredFull model](https://github.com/lkytal/PredFull) as per its [paper](https://pubs.acs.org/doi/10.1021/acs.analchem.9b04867).
The model provided by the authors in their repository (linked above) differs from the paper, so the former is used (I was not able to get a working model when attempting to transcribe the paper's model literally: the model simply overfits after reaching 0.20 cosine similarity on the validation set).

## Preparations

`pip install -r requirements.txt`

## Reproduction

- Download data from [the PredFull website](https://www.predfull.com/datasets)
- Run `predfull_data_to_h5.py infile outfile` where `outfile` to preprocess the mgf file and save is as hdf5
- [Optional] Download the `pm.h5` pretrained model from [the PredFull repository](https://github.com/lkytal/PredFull)
- Set `DATA`, `PRETRAINED`, `USE_PRETRAINED`, `RESET_PRETRAINED` and `FMT` parameters in the `predfull_keras.py` script as desired as per the instructions in the script
- Run `predfull_keras.py PHASE` where `PHASE` is the training phase number. To reproduce the paper, that's `python predfull_keras.py 0 && python predfull_keras.py 1 && python predfull_keras.py 2`

Omitting the `PHASE` parameter just trains the model endlessly with a set learning rate, it's meant to be customized as needed or used as an examnple for something more serious.

## Current Status
Model summary is identical between the model from the repository and this model, but this model fails to reach the same performance as a retrained pretrained model.
