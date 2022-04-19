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
import logging
import numpy as np
import os

# A hack to get OpenMM and PyTorch to peacefully coexist
os.environ["OPENMM_DEFAULT_PLATFORM"] = "OpenCL"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import torch
import json

from ipa_refine.config import model_config
from ipa_refine.model.refinement_model import RefinementModel
from ipa_refine.data.data_modules import RefinementDataModule
from ipa_refine.data.pipeline import RefinementDataPipeline
from ipa_refine.data.feature_pipeline import FeaturePipeline
from ipa_refine.data.refinement_features import get_features
from ipa_refine.np.protein import Protein, to_pdb
from ipa_refine.utils.loss import compute_drmsd
from ipa_refine.old_scripts.eval_drmsd import eval_drmsd, compare_drmsd, eval_drmsd_loss_batch2, eval_drmsd_loss_reference


execution_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/2022-01-07 16:39:23'
model_weights = 'model_weights'

data_path = '/home/nn122/remote_runtime/openfold/openfold/data/mini_dataset.json'
f = open(data_path, 'r')
mapping = json.load(f)
f.close()
random.shuffle(mapping)


def save_pred_pdb(aatype, positions, b_factors, type):
    # index = torch.tensor([0,1,2], dtype=torch.long)
    # pos = torch.index_select(positions, dim=1, index=index)
    object = Protein(
        atom_positions=positions,
        aatype=aatype,
        atom_mask=np.ones((len(aatype), 3)),
        residue_index=np.arange(0, len(aatype)),
        b_factors=b_factors
    )
    pdb_str = to_pdb(object)
    with open(os.path.join(execution_path, str(type)+'.pdb'), 'w') as f:
        f.write(pdb_str)


def main(args):

    config = model_config(
        "model_1",
        train=True,
        low_prec=True
    )
    model = RefinementModel(config, **config.model["structure_module"])
    weights_dict = torch.load(os.path.join(execution_path, model_weights))
    new_weights_dict = {}
    for key in weights_dict:
        new_key = key[6:]
        new_weights_dict[new_key] = weights_dict[key]
    model.load_state_dict(new_weights_dict)
    # model.load_state_dict(torch.load(os.path.join(execution_path, model_weights)))
    model = model.eval()
    model = model.to(device='cuda')

    logging.info("Generating features...")
    # data_module = RefinementDataModule(
    #     config=config.data,
    #     train=False,
    #     batch_seed=args.seed
    # )
    # data_module.prepare_data()
    # data_module.setup()

    data_pipeline = RefinementDataPipeline()
    feature_pipeline = FeaturePipeline(config.data)

    test_sample = random.sample(mapping, 50)
    initial_rmsd = np.zeros_like(test_sample)
    new_rmsd = np.zeros_like(test_sample)
    index = torch.tensor([1], dtype=torch.long)
    print("Executing model...")

    for i, sample in enumerate(test_sample):
        processed_feature_dict = {'x_features': get_features(sample['x'])}
        ref_features = data_pipeline.process_pdb(pdb_path=sample['x'])
        ref_processed_feature_dict = feature_pipeline.process_features(ref_features)

        gt_features = data_pipeline.process_pdb(pdb_path=sample['y'])
        gt_processed_feature_dict = feature_pipeline.process_features(gt_features)

        ref_coordinates = ref_processed_feature_dict['all_atom_positions'][None, :, 1, :]
        gt_coordinates = gt_processed_feature_dict['all_atom_positions'][None, :, 1, :]
        initial_rmsd[i] = eval_drmsd_loss_batch2(ref_coordinates, gt_coordinates)
        batch = processed_feature_dict
        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device='cuda')
                for k, v in batch['x_features'].items()
            }
            batch = {'x_features': batch}
            out = model(batch)

        pred_coordinates = out['sm']["positions"][-1][None, :, 1, :]
        # aatype = batch['x_features']['aatype']
        # b_factors = np.repeat(out['plddt'][..., None].detach().numpy(), 3, axis=-1)
        pred_coordinates = pred_coordinates.cpu()
        new_rmsd[i] = eval_drmsd_loss_batch2(pred_coordinates, gt_coordinates)
        print('loss = ', initial_rmsd[i], ' to ', new_rmsd[i])
        # save_pred_pdb(aatype, ref_coordinates, b_factors, 'initial_refined')
        # save_pred_pdb(aatype, gt_coordinates, b_factors, 'ground_truth')
        # save_pred_pdb(aatype, out['sm']["positions"][-1], b_factors, 'predicted')

    print('\nMean inital= ', initial_rmsd.mean())
    print('\nMean new= ', new_rmsd.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    main(args)
