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
from ipa_refine.old_scripts.eval_drmsd import eval_drmsd, compare_drmsd, eval_drmsd_loss_batch, eval_drmsd_loss_reference
from ipa_refine.np.protein import Protein, to_pdb

weights_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/2021-12-10 01:32:39/model_weights'
sample_path = {
    'x': '/home/nn122/Data4GNNRefine/CASP12/T0880/FALCON_TOPOX_TS3.pdb',
    'y': '/home/nn122/Data4GNNRefine/Native/CASP12/T0880.pdb'
}
data_path = '/home/nn122/remote_runtime/openfold/openfold/data/mini_dataset.json'
pdb_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/test_pred.pdb'
f = open(data_path, 'r')
mapping = json.load(f)
f.close()
random.shuffle(mapping)


def save_pred_pdb(outputs, batch):
    index = torch.tensor(np.arange(0,14), dtype=torch.long, device=batch['x_features']['aatype'].device)
    pred_atoms = outputs['sm']["positions"][-1]
    pos = torch.index_select(pred_atoms,dim=1,index=index)
    object = Protein(
        atom_positions=pos,
        aatype=batch['x_features']['aatype'],
        atom_mask=np.ones((len(batch['x_features']['aatype']), 3)),
        residue_index=np.arange(0, len(batch['x_features']['aatype'])),
        b_factors=np.repeat(outputs['plddt'][..., None].detach().numpy(), 3, axis=-1)
    )
    pdb_str = to_pdb(object)
    with open(pdb_path, 'w') as f:
        f.write(pdb_str)


# def save_pred_pdb_str(coordinates):
#     index = torch.tensor(np.arange(0,14), dtype=torch.long, device=batch['x_features']['aatype'].device)
#     pred_atoms = outputs['sm']["positions"][-1]
#     pos = torch.index_select(pred_atoms,dim=1,index=index)
#     object = Protein(
#         atom_positions=pos,
#         aatype=batch['x_features']['aatype'],
#         atom_mask=np.ones((len(batch['x_features']['aatype']), 3)),
#         residue_index=np.arange(0, len(batch['x_features']['aatype'])),
#         b_factors=np.repeat(outputs['plddt'][..., None].detach().numpy(), 3, axis=-1)
#     )
#     pdb_str = to_pdb(object)
#     with open(pdb_path, 'w') as f:
#         f.write(pdb_str)


def main(args):

    config = model_config(
        "model_1",
        train=True,
        low_prec=True
    )
    model = RefinementModel(config, **config.model["structure_module"])
    model.load_state_dict(torch.load(weights_path))
    model = model.eval()
    # # import_jax_weights_(model, args.param_path)
    # model = model.to(args.model_device)

    # train_len = int(0.8 * len(mapping))
    # test_len = len(mapping) - train_len
    # val_len = int(0.2 * train_len)
    # train_len = train_len - val_len
    # train_paths, test_paths = random_split(mapping, [train_len + val_len, test_len])
    # train_paths, val_paths = random_split(train_paths, [train_len, val_len])

    logging.info("Generating features...")

    data_pipeline = RefinementDataPipeline()
    feature_pipeline = FeaturePipeline(config.data)

    test_sample = random.sample(mapping, 1)
    drmsd = np.zeros(len(test_sample))

    for i, sample in enumerate(test_sample):

        processed_feature_dict = {'x_features': get_features(sample['x'])}
        gt_features = data_pipeline.process_pdb(pdb_path=sample['y'])
        gt_processed_feature_dict = feature_pipeline.process_features(gt_features)

        # Saving ground truth processed structure
        # coordinates = gt_processed_feature_dict['atom14_gt_positions'][0,:,]

        ref_features = data_pipeline.process_pdb(pdb_path=sample['x'])
        ref_processed_feature_dict = feature_pipeline.process_features(ref_features)

        # logging.info("Executing model...")
        batch = processed_feature_dict
        # with torch.no_grad():
        #     # batch = {
        #     #     k: torch.as_tensor(v, device=args.model_device)
        #     #     for k, v in batch.items()
        #     # }
        #
        t = time.time()
        out = model(batch)
        logging.info(f"Inference time: {time.time() - t}")
        # drmsd[i] = eval_drmsd(out["sm"], gt_processed_feature_dict)
        # save_pred_pdb(out, batch)
        drmsd[i] = compare_drmsd(ref_processed_feature_dict, gt_processed_feature_dict)

    print('Average DRMSD for test set of size (', len(test_sample), ') = ', drmsd.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "fasta_path", type=str,
    # )
    # add_data_args(parser)
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
        required=True
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--model_name", type=str, default="model_1",
        help="""Name of a model config. Choose one of model_{1-5} or 
             model_{1-5}_ptm, as defined on the AlphaFold GitHub."""
    )
    parser.add_argument(
        "--param_path", type=str, default=None,
        help="""Path to model parameters. If None, parameters are selected
             automatically according to the model name from 
             openfold/resources/params"""
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        '--preset', type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        '--data_random_seed', type=str, default=None
    )

    args = parser.parse_args()

    if (args.param_path is None):
        args.param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.model_name + ".npz"
        )

    if (args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    # if (args.bfd_database_path is None and
    #         args.small_bfd_database_path is None):
    #     raise ValueError(
    #         "At least one of --bfd_database_path or --small_bfd_database_path"
    #         "must be specified"
    #     )

    main(args)
