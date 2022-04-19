# GNNRefine

**Fast and effective protein model refinement by deep graph neural networks**

This package is the source code of GNNRefine, which is a fast and effective GNN (graph neural networks) based method for protein model refinement with very limited conformation sampling. 

## Requirements
* python >=3.6
* pytorch >=1.1.0
* dgl >=0.4.3
* biopython >=1.75
* dssp or mkdssp (called by biopython, see https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html)
* pyrosetta >=2019.35
* numpy >=1.18.1

## Usage

### Download package
The code of GNNRefine is available at http://raptorx.uchicago.edu

### Prepare starting model
The input of GNNRefine is the PDB file of starting model.

Please note that the residue indexs in the PDB file should be consecutive and start from 1. 

### Run
```
$ python GNNRefine.py -h
usage: GNNRefine.py [-h] [-n_decoy N_DECOY] [-n_proc N_PROC] [-device_id DEVICE_ID]
                    [-save_qa] [-save_le_decoy] [-save_all_decoy] [-only_pred_dist]
                    input output

positional arguments:
  input                 path of starting model or folder containing starting models.
  output                path of output folder.

optional arguments:
  -h, --help            show this help message and exit
  -n_decoy N_DECOY      number of decoys built in each iteration. (default: 1)
  -n_proc N_PROC        number of processes running in parallel (>=1, recommendation is >=n_decoy). (default: 1)
  -device_id DEVICE_ID  device id (-1 for CPU, >=0 for GPU). (default: -1)
  
  -save_qa              save QA results. (default: False)
  -save_le_decoy        save decoy models of lowest erergy in each iteration. (default: False)
  -save_all_decoy       save all decoy models. (default: False)
  -only_pred_dist       only predict the refined distance probability distribution. (default: False)
```

Suppose you have a starting model at `example/R0974s1.pdb`, you can run the following command to generate the refined model.

```
python GNNRefine.py example/R0974s1.pdb output/
```

The refined model will be saved at `output/R0974s1.pdb.refined.pdb`.

Examples of the starting models (and their refined models and QA results) for R0974s1, R0976-D2, R0993s2 and R1082 can be found at `example/`.

GNNRefine can also output related files with the following optional arguments:
* `-save_qa`: save the global and local quality estimations for the refined models (based on lDDT).
* `-save_le_decoy`: save decoy models of lowest erergy in each iteration.
* `-save_all_decoy`: save all decoy models generated in the whole refinement process.
* `-only_pred_dist`: predict refined distance probability distribution for the starting model.


## Reference
Xiaoyang Jing, Jinbo Xu. Fast and effective protein model refinement by deep graph neural networks. bioRxiv 2020.12.10.419994 (2020) [doi:10.1101/2020.12.10.419994](https://doi.org/10.1101/2020.12.10.419994).

## Contact
Xiaoyang Jing, xyjing@ttic.edu

Jinbo Xu, jinboxu@gmail.com