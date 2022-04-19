'''
Create Train / Test split for validated data and write to file.
Dataset for training is stored in Json format as -> [{x: "path to refined pdb", y: "path to ground truth file"}]
'''


import argparse
import logging
import pandas as pd
from os.path import join
import json
import random
from ipa_refine.utils import refinement_utils as Utils


def get_paths(data_path, e):
    ref = e[3]
    if '.pdb' not in e[3]:
        ref = e[3]+'.pdb'
    x = join(data_path, e[1], e[2], ref)
    y = join(data_path, 'Native', e[1], e[2]+'.pdb')
    return {'x': x, 'y': y}


def check_seq_match(sample, seq_len_limit):
    try:
        gt_seq = Utils.get_seqs_from_pdb(sample['x'])
        ref_seq = Utils.get_seqs_from_pdb(sample['y'])
        if gt_seq == ref_seq and len(gt_seq) < seq_len_limit:
            return True
        else:
            return False
    except:
        return False


def main(args):
    df = pd.read_csv(args.train_file, sep=' ', names=('dataset', 'protein', 'refinement'))
    dataset = []
    for row in df.itertuples():
        temp = get_paths(args.dataset_path, row)
        if temp is not None:
            dataset.append(get_paths(args.dataset_path, row))
    random.shuffle(dataset)
    dataset = dataset[:args.data_size_limit]
    # Filter out longer and unmatching sequences
    dataset = [sample for sample in dataset if check_seq_match(sample, args.seq_len_limit)]
    test_len = int(0.1 * len(dataset))
    train_set = dataset[:-(2*test_len)]
    val_set = dataset[-(2*test_len):-test_len]
    test_set = dataset[-test_len:]
    data = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }
    with open(args.file_to, 'w') as convert_file:
        convert_file.write(json.dumps(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_size_limit", type=int, default=20000,
        help="""Number of training examples to use to create filtered data_list for training"""
    )
    parser.add_argument(
        "--seq_len_limit", type=int, default=200,
        help="""Consider protein sequences upto residue length."""
    )
    parser.add_argument(
        "--train_file", type=str, default="/home/nn122/remote_runtime/newfold/ipa_refine/LISTS/Train.list.1",
        help="""File path for saving filtered data_list"""
    )
    parser.add_argument(
        "--file_to", type=str, default="/home/nn122/remote_runtime/newfold/ipa_refine/LISTS/clean_train_list.json",
        help="""File path for saving filtered data_list"""
    )
    parser.add_argument(
        "--dataset_path", type=str, default="/home/nn122/Data4GNNRefine",
        help="""File path for saving filtered data_list"""
    )
    args = parser.parse_args()
    main(args)
