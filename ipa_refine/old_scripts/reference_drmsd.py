# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from datetime import date
import logging
import numpy as np
import os

# A hack to get OpenMM and PyTorch to peacefully coexist
os.environ["OPENMM_DEFAULT_PLATFORM"] = "OpenCL"

import pickle
import random
import sys
import time
import torch
import json
# import tensor
from torch.utils.data import RandomSampler, Dataset, random_split

from ipa_refine.config import model_config
from ipa_refine.data import feature_pipeline, data_pipeline
from ipa_refine.data.data_modules import RefinementSingleDataset
from ipa_refine.model.refinement_model import RefinementModel
from ipa_refine.data.pipeline import RefinementDataPipeline
from ipa_refine.data.feature_pipeline import FeaturePipeline
from ipa_refine.np import residue_constants, protein
import ipa_refine.np.relax.relax as relax
from ipa_refine.utils.import_weights import (
    import_jax_weights_,
)
from ipa_refine.data.refinement_features import get_features
from ipa_refine.utils.tensor_utils import (
    tensor_tree_map,
)
from ipa_refine.old_scripts.eval_drmsd import eval_drmsd, compare_drmsd, eval_drmsd_loss_batch
from ipa_refine.np.protein import Protein, to_pdb

weights_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/2021-12-10 01:32:39/model_weights'
sample_path = {
    'x': '/home/nn122/Data4GNNRefine/CASP12/T0880/FALCON_TOPOX_TS3.pdb',
    'y': '/home/nn122/Data4GNNRefine/Native/CASP12/T0880.pdb'
}
mini_train_list = '/home/nn122/remote_runtime/newfold/ipa_refine/old_scripts/mini_train_list.json'
f = open(mini_train_list, 'r')
mapping = json.load(f)
f.close()
mapping = mapping["train"]
random.shuffle(mapping)
# mapping = mapping[:400]

config = model_config(
    "model_1",
    train=True,
    low_prec=True
)

data_pipeline = RefinementDataPipeline()
feature_pipeline = FeaturePipeline(config.data)

drmsd = np.zeros_like(mapping)

for i, sample in enumerate(mapping):
    gt_features = data_pipeline.process_pdb(pdb_path=sample['y'])
    gt_processed_feature_dict = feature_pipeline.process_features(gt_features)
    ref_features = data_pipeline.process_pdb(pdb_path=sample['x'])
    ref_processed_feature_dict = feature_pipeline.process_features(ref_features)
    drmsd[i] = float(compare_drmsd(ref_processed_feature_dict, gt_processed_feature_dict))

print('AVG DRMSD:', drmsd.mean())