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


feat_class = {
    'seq': {
        'node': ['onehot', 'rPosition'],     #['onehot', 'rPosition'],
        'edge': ['SepEnc']
    },
    'struc': {
        'node': ['SS3', 'RSA', 'Dihedral'],
        'edge': ['Ca1-Ca2', 'Cb1-Cb2', 'N1-O2', 'Ca1-Cb1-Cb2',
                        'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2']
    }
}


def get_seq_feature(pdb_file):
    seq = Utils.get_seqs_from_pdb(pdb_file)
    # node_feat
    node_feat = {
        'onehot': Utils.get_seq_onehot(seq),
        'rPosition': Utils.get_rPos(seq),
    }
    # edge_feat
    edge_feat = {
        'SepEnc': Utils.get_SepEnc(seq),
    }
    return node_feat, edge_feat, len(seq)


def get_struc_feat(pdb_file, seq_len):
    # node feat
    node_feat = Utils.get_DSSP_label(pdb_file, [1, seq_len])
    # atom_emb
    embedding, frame_atoms, res_names = Utils.get_atom_emb(pdb_file, [1, seq_len])
    node_feat['atom_emb'] = {
        'embedding': embedding.astype(np.float32),
    }
    # edge feat
    edge_feat = Utils.calc_geometry_maps(pdb_file, [1, seq_len], feat_class['struc']['edge'])
    return node_feat, edge_feat, frame_atoms, res_names


def get_features(pdb_file):

    with open(pdb_file, 'r') as f:
        pdb_str = f.read()
    ref_protein_object = protein.from_pdb_string(pdb_str)
    aatype = ref_protein_object.aatype

    feature = {"node": None, "edge": None}

    # seq feature
    seq_node_feat, seq_edge_feat, seq_len = get_seq_feature(pdb_file)
    for _feat in feat_class['seq']['node']:
        feature['node'] = seq_node_feat[_feat] if feature['node'] is None else np.concatenate(
            (feature['node'], seq_node_feat[_feat]), axis=-1)
    for _feat in feat_class['seq']['edge']:
        feature['edge'] = seq_edge_feat[_feat] if feature['edge'] is None else np.concatenate(
            (feature['edge'], seq_edge_feat[_feat]), axis=-1)

    # Structure feature
    struc_node_feat, struc_edge_feat, frame_atoms, res_names = get_struc_feat(pdb_file, seq_len)
    for _feat in feat_class['struc']['node']:
        feature['node'] = struc_node_feat[_feat] if feature['node'] is None else np.concatenate(
            (feature['node'], struc_node_feat[_feat]), axis=-1)
    for _feat in feat_class['struc']['edge']:
        feature['edge'] = struc_edge_feat[_feat] if feature['edge'] is None else np.concatenate(
            (feature['edge'], struc_edge_feat[_feat]), axis=-1)

    frames = [Utils.rigidFrom3Points(x) for i, x in enumerate(frame_atoms)]
    R = [f[0] for i, f in enumerate(frames)]
    t = [f[1] for i, f in enumerate(frames)]

    # feature
    feature_tensor = np.nan_to_num(feature)
    feature_tensor['node'] = torch.tensor(feature['node'].astype(np.float32))
    feature_tensor['edge'] = torch.tensor(feature['edge'].astype(np.float32))
    # feature['adj'] = adj.astype(np.float32)
    # feature['atom_emb'] = struc_node_feat['atom_emb']
    feature_tensor['frames_R'] = torch.tensor(np.asarray(R).astype(np.float32))
    feature_tensor['frames_t'] = torch.tensor(np.asarray(t).astype(np.float32))
    feature_tensor['aatype'] = torch.tensor(np.asarray(aatype).astype(np.float32))

    return feature_tensor