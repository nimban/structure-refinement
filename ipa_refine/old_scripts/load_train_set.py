import pandas as pd
from os.path import join
import json
import random
from ipa_refine.utils import refinement_utils as Utils

train_list = '/home/nn122/remote_runtime/newfold/ipa_refine/old_scripts/train_list'
mini_train_list = '/home/nn122/remote_runtime/newfold/ipa_refine/old_scripts/mini_train_list2.json'
base_path = '/data/tc207/Data4GNNRefine'


def get_paths(e):
    ref = e[3]
    if '.pdb' not in e[3]:
        ref = e[3]+'.pdb'
    x = join(base_path, e[1], e[2], ref)
    y = join(base_path, 'Native', e[1], e[2]+'.pdb')
    return {'x': x, 'y': y}


def get_train_set(debug=False):
    df = pd.read_csv(train_list, sep='/', names=('dataset', 'protein', 'refinement'))
    train_set = []
    for row in df.itertuples():
        temp = get_paths(row)
        if temp is not None:
            train_set.append(get_paths(row))
    if debug:
        random.shuffle(train_set)
        train_set = train_set[:4000]
    return train_set


# The Function Used in Dataloader
def get_filtered_train_set(debug=False, limit=-1):
    f = open(mini_train_list, 'r')
    mapping = json.load(f)
    f.close()
    mapping = mapping["train"]
    random.shuffle(mapping)
    mapping = mapping[:limit]
    return mapping


def check_seq_match(sample):
    try:
        gt_seq = Utils.get_seqs_from_pdb(sample['x'])
        ref_seq = Utils.get_seqs_from_pdb(sample['y'])
        if gt_seq == ref_seq and len(gt_seq) < 200:
            return True
        else:
            return False
    except:
        return False


def save_filtered_mini_dataset():
    df = pd.read_csv(train_list, sep='/', names=('dataset', 'protein', 'refinement'))
    dataset = []
    for row in df.itertuples():
        temp = get_paths(row)
        if temp is not None:
            dataset.append(get_paths(row))
    random.shuffle(dataset)
    dataset = dataset[:20000]
    # Filter out longer and unmatching sequences
    dataset = [sample for sample in dataset if check_seq_match(sample)]
    test_len = int(0.1 * len(dataset))
    train_set = dataset[:-(2*test_len)]
    val_set = dataset[-(2*test_len):-test_len]
    test_set = dataset[-test_len:]
    data = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }
    # with open('mini_train_list.json', 'w') as convert_file:
    #     convert_file.write(json.dumps(data))


# save_filtered_mini_dataset()