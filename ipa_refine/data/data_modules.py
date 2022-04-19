import copy
from functools import partial
import json
import logging
import os
import json
import pickle
from typing import Optional, Sequence
import random

import ml_collections as mlc
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.functional import pad
from torch.utils.data import RandomSampler, Dataset, random_split
from ipa_refine.utils import refinement_utils as Utils
from ipa_refine.data.pipeline import RefinementDataPipeline
from ipa_refine.data.feature_pipeline import FeaturePipeline
from ipa_refine.np import residue_constants, protein
from ipa_refine.utils.tensor_utils import tensor_tree_map, dict_multimap
from ipa_refine.data.refinement_features import get_features


# def save_split(train, val, test):
#     split_dict = {
#         'train_paths': train,
#         'val_paths': val,
#         'test_paths': test
#     }
#     with open(split_path, 'w') as convert_file:
#         convert_file.write(json.dumps(split_dict))


# The Function Used in Dataloader
def get_filtered_train_set(data_list_file, limit=-1):
    f = open(data_list_file, 'r')
    mapping = json.load(f)
    f.close()
    mapping = mapping["train"]
    random.shuffle(mapping)
    mapping = mapping[:limit]
    return mapping


class RefinementSingleDataset(Dataset):

    def __init__(self,
                 data_paths: [],
                 mode: str,
                 config: mlc.ConfigDict,
                 adj_type='Cb1-Cb2',
                 adj_cutoff=10,
                 ):

        self.config = config
        self.adj_type = adj_type
        self.adj_cutoff = adj_cutoff
        self.mapping = data_paths
        self.data_len = len(self.mapping)
        self.data_pipeline = RefinementDataPipeline()
        self.feature_pipeline = FeaturePipeline(self.config)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        ref_pdb_file = self.mapping[idx]['x']
        gt_pdb_file = self.mapping[idx]['y']
        feature_tensor = get_features(ref_pdb_file)
        gt_features = self.data_pipeline.process_pdb(pdb_path=gt_pdb_file)
        gt_feats = self.feature_pipeline.process_features(gt_features)
        sample = {
            'x_features': feature_tensor,
            'y_features': gt_feats
        }
        return sample


class RefinementBatchCollator:

    def __init__(self, config, generator, stage="train"):
        self.config = config
        self.generator = generator
        self.stage = stage

    def __call__(self, raw_batch):
        processed_batch = []
        max_len = max([len(p['x_features']['node']) for p in raw_batch])
        for prot in raw_batch:
            processed_prot = {}
            seq_size = len(prot['x_features']['node'])
            pad_size = max_len - seq_size
            # for x and y feats
            for key_out in prot.keys():
                new_prot = {}
                # for each individual feat
                for key in prot[key_out].keys():
                    feature = prot[key_out][key]
                    shape = [dim == seq_size for dim in feature.size()]
                    pad_shape = [(0, pad_size*d)for d in shape]
                    pad_shape.reverse()
                    pad_shape = [item for sublist in pad_shape for item in sublist]
                    newfeat = pad(feature, pad_shape)
                    new_prot[key] = newfeat
                processed_prot[key_out] = new_prot
            processed_batch.append(processed_prot)

        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, processed_batch)


class RefinementDataModule(pl.LightningDataModule):

    def __init__(self,
                 config: mlc.ConfigDict,
                 data_list_file: str,
                 # eval: bool,
                 train_limit: Optional[int],
                 batch_seed: Optional[int] = None
                 ):
        super(RefinementDataModule, self).__init__()

        self.config = config
        self.data_list_file = data_list_file
        self.batch_seed = batch_seed
        self.train_limit = train_limit
        self.eval = False

    def setup(self, stage: Optional[str] = None):
        # if (stage is None):
        #     stage = "train"
        stage = "train"

        mapping = get_filtered_train_set(self.data_list_file, self.train_limit)  #get_train_set()

        # Most of the arguments are the same for the three datasets
        # dataset_gen = partial(RefinementSingleDataset)
        test_len = int(0.1 * len(mapping))
        val_len = int(0.12 * (len(mapping) - test_len))
        train_len = len(mapping) - test_len - val_len
        self.train_paths, self.test_paths = random_split(mapping, [train_len+val_len, test_len])
        self.train_paths, self.val_paths = random_split(self.train_paths, [train_len, val_len])

        ## TODO: Function to write split to disk with model_ID
        # save_split(self.train_paths, self.val_paths, self.test_paths)


        if (stage == 'train'):  #self.training_mode):
            self.train_dataset = RefinementSingleDataset(self.train_paths, 'train', self.config)
            self.val_dataset = RefinementSingleDataset(self.val_paths, 'eval', self.config)
        else:
            self.predict_dataset = RefinementSingleDataset(self.test_paths, mode='predict')

    def _gen_batch_collator(self, stage):
        """ We want each process to use the same batch collation seed """
        generator = torch.Generator()
        if (self.batch_seed is not None):
            generator = generator.manual_seed(self.batch_seed)
        collate_fn = RefinementBatchCollator(
            self.config, generator, stage
        )
        return collate_fn

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=self._gen_batch_collator("train"),
        )

    def val_dataloader(self):
        if (self.val_dataset is not None):
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.data_module.data_loaders.batch_size,
                num_workers=self.config.data_module.data_loaders.num_workers,
                collate_fn=self._gen_batch_collator("eval")
            )
        return None

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=self._gen_batch_collator("predict")
        )