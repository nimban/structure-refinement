# {"query":{"type:"terminal","service"sequence}, return_type:entry}

#{"query": {"type": "terminal","service": "sequence","parameters": { "evalue_cutoff": 1, "identity_cutoff": 0.9,"target": "pdb_protein_sequence","value": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLPARTVETRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMNCKCVIS"}},"request_options": {"scoring_strategy": "sequence"}, "return_type": "entry"}

import json
import os
import csv
import pandas as pd
import rough
import numpy as np
import requests
from ipa_refine.utils import refinement_utils

# url = 'https://search.rcsb.org/rcsbsearch/v1/query?'
# with open("/home/nn122/remote_runtime/openfold/openfold/data/pdb_search.json", "r") as read_file:
#     req_data = json.load(read_file)

pdb_id_dict = {}
df = pd.read_csv('targetlist.csv', sep=';')

print(df.head())

# base_path = '/home/nn122/Data4GNNRefine/Native/CASP13/'
# data_path = '/home/nn122/Data4GNNRefine/CASP13/'
#
# native_files = [pth.split('.')[0] for pth in os.listdir(base_path)]
# mapping = []
# for f in native_files:
#     preds = [p for p in os.listdir(data_path+f)]
#     mapping.append({
#         'protein': f,
#         'gt_path': base_path + f + '.pdb',
#         'refined_paths': [data_path + f + '/' + pred for pred in preds]
#     })
#
# def get_pdb_id(path):
#     try:
#         protein_seq = Utils.get_seqs_from_pdb(path)
#     except:
#         return False
#     req_data["query"]["parameters"]["value"] = protein_seq
#     response = requests.post(url, json=req_data)
#     if response.status_code == 200:
#         res = response.json()
#         if len(res["result_set"]) < 2:
#             return res["result_set"][0]["identifier"]
#         else:
#             return False
#     else:
#         return False
#
#
# # Get IDs for one refined of each type
#
# failed_count = 0
# match_count = 0
# protein_ids = []
#
# for protein in mapping:
#     res = get_pdb_id(protein["refined_paths"][0])
#     if res == False:
#         failed_count += 1
#         gt_seq = Utils.get_seqs_from_pdb(protein['gt_path'])
#         ref_seq = Utils.get_seqs_from_pdb(protein['refined_paths'][0])
#         if gt_seq == ref_seq:
#             match_count +=1
#     else:
#         protein["protein_id"] = res
#         protein_ids.append(res)
#
# print('Failed: ', failed_count)
# print('Count not in PDB but gt file matches: ', match_count)
#
# textfile = open("list_file.txt", "w")
# for element in protein_ids:
#     textfile.write(element + ",")
# textfile.close()