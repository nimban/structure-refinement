# Refinement using IPA Structure Module

## Overview

Test AlphaFold inspired structure module using Invariant Point Attention mechanism for learning to refine Protein Structures.
Training/Testing against CASP datasets.


## Setup 

Dataset in Data4GNNRefine Folder. (/home/nn122/Data4GNNRefine) on Box 4.

All pdb files must have .pdb extension for compatibility with DSSP library.

Setup environment using installdependencies and activate conda scripts from open fold. 
(Can load existing environment on Box 4 by running )
```bash
cd /home/nn122/openfold
source install_third_party_dependencies.sh
source activate_conda_env.sh
```

All Following scripts are run from newfold folder.

To create a filtered list of examples for training based on seq length, run - 
```bash
python3 create_training_list.py
```


## Training 

For training a model - 
```bash
python3 training.py
```


## Evaluation

For predicting a protein structure using a pretrainined model - 
```bash
python3 inference.py
```

For Evaluating the performance of an existing model -
```bash
python3 evaluation.py
```

## Previous Experiments

WandB Link - https://wandb.ai/nimban/ipa_refine_tuning
