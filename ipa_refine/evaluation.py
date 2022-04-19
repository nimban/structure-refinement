'''
Evaluate performance of a pretrained model against specified dataset
'''

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
from ipa_refine.data.pipeline import RefinementDataPipeline
from ipa_refine.data.feature_pipeline import FeaturePipeline
from ipa_refine.data.refinement_features import get_features
from ipa_refine.utils.loss import eval_drmsd_loss_batch


execution_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/2022-01-07 16:39:23'
model_weights = 'model_weights'

data_path = '/home/nn122/remote_runtime/openfold/openfold/data/mini_dataset.json'
f = open(data_path, 'r')
mapping = json.load(f)
f.close()
random.shuffle(mapping)


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
    model = model.eval()
    model = model.to(device='cuda')

    logging.info("Generating features...")
    data_pipeline = RefinementDataPipeline()
    feature_pipeline = FeaturePipeline(config.data)

    test_samples = random.sample(mapping, 50)
    initial_rmsd = np.zeros_like(test_samples)
    new_rmsd = np.zeros_like(test_samples)

    print("Executing model...")

    for i, sample in enumerate(test_samples):
        processed_feature_dict = {'x_features': get_features(sample['x'])}
        pred_features = data_pipeline.process_pdb(pdb_path=sample['x'])
        pred_processed_feature_dict = feature_pipeline.process_features(pred_features)

        gt_features = data_pipeline.process_pdb(pdb_path=sample['y'])
        gt_processed_feature_dict = feature_pipeline.process_features(gt_features)

        pred_coordinates = pred_processed_feature_dict['all_atom_positions'][None, :, 1, :]
        gt_coordinates = gt_processed_feature_dict['all_atom_positions'][None, :, 1, :]
        initial_rmsd[i] = eval_drmsd_loss_batch(pred_coordinates, gt_coordinates)
        batch = processed_feature_dict

        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device='cuda')
                for k, v in batch['x_features'].items()
            }
            batch = {'x_features': batch}
            out = model(batch)

        refined_coordinates = out['sm']["positions"][-1][None, :, 1, :]
        refined_coordinates = refined_coordinates.cpu()
        new_rmsd[i] = eval_drmsd_loss_batch(refined_coordinates, gt_coordinates)
        print('loss = ', initial_rmsd[i], ' to ', new_rmsd[i])

    print('\nMean Initial= ', initial_rmsd.mean())
    print('\nMean New= ', new_rmsd.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_device", type=str, default="gpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )

    args = parser.parse_args()

    if (args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)
