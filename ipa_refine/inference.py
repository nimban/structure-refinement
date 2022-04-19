'''
Use pretrained model to refine a protein structure and save PDB.
'''

import argparse
import logging
import numpy as np
import os

# A hack to get OpenMM and PyTorch to peacefully coexist
os.environ["OPENMM_DEFAULT_PLATFORM"] = "OpenCL"

import random
import torch
import json

from ipa_refine.config import model_config
from ipa_refine.model.refinement_model import RefinementModel
from ipa_refine.data.pipeline import RefinementDataPipeline
from ipa_refine.data.feature_pipeline import FeaturePipeline
from ipa_refine.data.refinement_features import get_features
from ipa_refine.np.protein import Protein, to_pdb

# File Path to load pretrained model weights from
execution_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/2022-01-03 17:26:25'
model_weights = 'model_weights'

# Load sample datapoint/protein structure to refine
data_path = '/home/nn122/remote_runtime/openfold/openfold/data/mini_dataset.json'
f = open(data_path, 'r')
mapping = json.load(f)
f.close()
random.shuffle(mapping)


# PDB files saved to folder with model_weights
def save_pred_pdb(aatype, positions, b_factors, type):
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
    model.load_state_dict(torch.load(os.path.join(execution_path, model_weights)))
    model = model.eval()

    logging.info("Generating features...")

    data_pipeline = RefinementDataPipeline()
    feature_pipeline = FeaturePipeline(config.data)

    # Pick sample protein for refinement
    test_sample = random.sample(mapping, 1)

    for i, sample in enumerate(test_sample):

        processed_feature_dict = {'x_features': get_features(sample['x'])}

        pred_features = data_pipeline.process_pdb(pdb_path=sample['x'])
        pred_processed_feature_dict = feature_pipeline.process_features(pred_features)

        gt_features = data_pipeline.process_pdb(pdb_path=sample['y'])
        gt_processed_feature_dict = feature_pipeline.process_features(gt_features)

        # Saving ground truth processed structure
        pred_coordinates = pred_processed_feature_dict['all_atom_positions']#[:, 1, :]
        gt_coordinates = gt_processed_feature_dict['all_atom_positions']#[:, 1, :]

        print("Executing model...")
        batch = processed_feature_dict
        with torch.no_grad():
            out = model(batch)

        aatype = batch['x_features']['aatype']
        b_factors = np.repeat(out['plddt'][..., None].detach().numpy(), 3, axis=-1)
        
        # Save Initial, Refined and Ground Truth PDB structures
        save_pred_pdb(aatype, pred_coordinates, b_factors, 'initial')
        save_pred_pdb(aatype, gt_coordinates, b_factors, 'ground_truth')
        save_pred_pdb(aatype, out['sm']["positions"][-1], b_factors, 'refined')

    print('Saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
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
