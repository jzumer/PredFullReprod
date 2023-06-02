# Reproducing the PredFull model

This repo contains code to reproduce the [PredFull model](https://github.com/lkytal/PredFull) as per its [paper](https://pubs.acs.org/doi/10.1021/acs.analchem.9b04867).
The model provided by the authors in their repository (linked above) differs from the paper, so the provided model is used as the basis for this reproduction.

## Preparations

`pip install -r requirements.txt`

## Reproduction

- Download data from [the PredFull website](https://www.predfull.com/datasets)
- Run `predfull_data_to_h5.py infile outfile` to preprocess the mgf file and save it in hdf5 format
- [Optional] Download the `pm.h5` pretrained model from [the PredFull repository](https://github.com/lkytal/PredFull)
- Set `DATA`, `PRETRAINED`, `USE_PRETRAINED`, `RESET_PRETRAINED` and `FMT` parameters in the `predfull_keras.py` script as desired as per the instructions in the script
- Run `predfull_keras.py PHASE` where `PHASE` is the training phase number. To reproduce the paper, that's `python predfull_keras.py 0 && python predfull_keras.py 1 && python predfull_keras.py 2`

Omitting the `PHASE` parameter just trains the model endlessly with a set learning rate, it's meant to be customized as needed or used as an example for something more serious.

## Current Status
Model `summary` is identical between the model from the repository and this model, and training proceeds similarly. Some activation functions may differ.
A script was provided by the paper's first author for reference for model creation, although it may not completely match the released model. It is available as `build_model_example.py`.
