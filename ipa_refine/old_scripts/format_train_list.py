# import pandas as pd
from os.path import join, exists
import json
# import random
# from ipa_refine.utils import refinement_utils as Utils

train_list = '/home/nn122/remote_runtime/newfold/ipa_refine/old_scripts/train_list'
mini_train_list = '/home/nn122/remote_runtime/newfold/ipa_refine/old_scripts/mini_train_list.json'
base_path = '/data/tc207/Data4GNNRefine'


# def get_paths(e):
#     ref = e[3]
#     if '.pdb' not in e[3]:
#         ref = e[3]+'.pdb'
#     x = join(base_path, e[1], e[2], ref)
#     y = join(base_path, 'Native', e[1], e[2]+'.pdb')
#     return {'x': x, 'y': y}
#
#
# def get_train_set(debug=False):
#     df = pd.read_csv(train_list, sep='/', names=('dataset', 'protein', 'refinement'))
#     train_set = []
#     for row in df.itertuples():
#         temp = get_paths(row)
#         if temp is not None:
#             train_set.append(get_paths(row))
#     if debug:
#         random.shuffle(train_set)
#         train_set = train_set[:4000]
#     return train_set
#
#
# def get_filtered_train_set(debug=False):
#     f = open(mini_train_list, 'r')
#     mapping = json.load(f)
#     f.close()
#     mapping = mapping["train"]
#     if debug:
#         random.shuffle(mapping)
#         mapping = mapping[:5000]
#     return mapping
#
#
# def check_seq_match(sample):
#     try:
#         gt_seq = Utils.get_seqs_from_pdb(sample['x'])
#         ref_seq = Utils.get_seqs_from_pdb(sample['y'])
#         if gt_seq == ref_seq and len(gt_seq) < 200:
#             return True
#         else:
#             return False
#     except:
#         return False
#
#
# def save_filtered_mini_dataset():
#     df = pd.read_csv(train_list, sep='/', names=('dataset', 'protein', 'refinement'))
#     dataset = []
#     for row in df.itertuples():
#         temp = get_paths(row)
#         if temp is not None:
#             dataset.append(get_paths(row))
#     random.shuffle(dataset)
#     dataset = dataset[:20000]
#     dataset = [sample for sample in dataset if check_seq_match(sample)]
#     test_len = int(0.1 * len(dataset))
#     train_set = dataset[:-(2*test_len)]
#     val_set = dataset[-(2*test_len):-test_len]
#     test_set = dataset[-test_len:]
#     data = {
#         "train": train_set,
#         "val": val_set,
#         "test": test_set,
#     }
#     # with open('mini_train_list.json', 'w') as convert_file:
#     #     convert_file.write(json.dumps(data))


def fix_paths(e):
    temp = {}
    for key in e.keys():
        if exists(e[key]):
            temp[key] = e[key]
        elif exists(e[key][:-4]):
            temp[key] = e[key][:-4]
        else:
            print('Error ', e[key])
    if 'x' not in temp or 'y' not in temp:
        print('wtf')
        return None
    else:
        return temp


# def format_pdb_file_extns():
#     f = open(mini_train_list, 'r')
#     mapping = json.load(f)
#     f.close()
#     data = {}
#     for key in mapping.keys():
#         data[key] = list(map(fix_paths, mapping[key]))
#     print(data.keys())
#     print(len(data['train']))
#     with open('/home/nn122/remote_runtime/newfold/ipa_refine/old_scripts/mini_train_list2.json', 'w') as convert_file:
#         convert_file.write(json.dumps(data))

def format_pdb_file_extns():
    f = open(mini_train_list, 'r')
    mapping = json.load(f)
    f.close()
    data = {
        "train": [],
        "val": [],
        "test": [],
    }
    for type in mapping.keys():
        for e in mapping[type]:
            temp = {}
            for key in e.keys():
                if exists(e[key]):
                    temp[key] = e[key]
                elif exists(e[key][:-4]):
                    temp[key] = e[key][:-4]
                else:
                    print('Error ', e[key])
            if 'x' not in temp or 'y' not in temp:
                print('wtf')
                return None
            else:
                data[type].append(temp)
    print(data.keys())
    print(len(data['train']))
    with open('/home/nn122/remote_runtime/newfold/ipa_refine/old_scripts/mini_train_list2.json', 'w') as convert_file:
        convert_file.write(json.dumps(data))


format_pdb_file_extns()

# print(exists('/data/tc207/Data4GNNRefine/CASP8/T0397/MULTICOM-CLUSTER_TS5'))