import json
import os
import random

from ipa_refine.utils import refinement_utils as Utils

base_path = '/home/nn122/Data4GNNRefine/Native/CASP13/'
data_path = '/home/nn122/Data4GNNRefine/CASP13/'

native_files = [pth.split('.')[0] for pth in os.listdir(base_path)]
mapping = []
for f in native_files:
    preds = [p for p in os.listdir(data_path+f)]
    mapping.append({
        'protein': f,
        'gt_path': base_path + f + '.pdb',
        'refined_paths': [data_path + f + '/' + pred for pred in preds]
    })

protein_id_dict = {}
filtered_proteins = []
count = 0
long_count = 0

for protein in mapping:
    # if len(filtered_proteins) > 100:
    #     break
    try:
        gt_seq = Utils.get_seqs_from_pdb(protein['gt_path'])
        if len(gt_seq) > 200:
            long_count +=1
            continue
    except:
        continue
    for ref_path in protein['refined_paths']:
        try:
            ref_seq = Utils.get_seqs_from_pdb(ref_path)
        except:
            continue
        if gt_seq == ref_seq:
            filtered_proteins.append({
                'x': ref_path,
                'y': protein['gt_path']
            })
        else:
            count +=1

print('Discarded Unmatch- ', count)
print('Discarded Long- ', long_count)

# random.shuffle(filtered_proteins)

with open('mini_dataset.json', 'w') as convert_file:
    convert_file.write(json.dumps(filtered_proteins))