import numpy as np
import torch
from ipa_refine.utils.loss import compute_drmsd


def eval_drmsd(outputs, batch):
    index = torch.tensor([1], dtype=torch.long, device=batch['aatype'].device)
    gt_atoms = batch["all_atom_positions"]
    gt_ca_atoms = torch.squeeze(torch.index_select(gt_atoms,dim=1,index=index), dim=1)
    pred_atoms = outputs["positions"][-1]
    pred_ca_atoms = torch.squeeze(torch.index_select(pred_atoms,dim=1,index=index), dim=1)
    res = compute_drmsd(pred_ca_atoms, gt_ca_atoms)
    return res


def compare_drmsd(ref_features, gt_features):
    gt_ca_atoms = gt_features["all_atom_positions"][..., :, 1, :]
    ref_ca_atoms = ref_features["all_atom_positions"][..., :, 1, :]
    drmsd = compute_drmsd(ref_ca_atoms, gt_ca_atoms)
    return drmsd


def eval_drmsd_loss_batch(outputs, batch):
    # index = torch.tensor([1], dtype=torch.long, device=batch['aatype'].device)
    # gt_atoms = batch["all_atom_positions"]
    # gt_ca_atoms = torch.squeeze(torch.index_select(gt_atoms,dim=2,index=index), dim=2)
    # pred_atoms = outputs['sm']["positions"][-1]
    # pred_ca_atoms = torch.squeeze(torch.index_select(pred_atoms,dim=2,index=index), dim=2)

    gt_ca_atoms = batch["all_atom_positions"][..., :, 1, :]
    pred_ca_atoms = outputs['sm']["positions"][-1][..., :, 1, :]
    drmsd = np.zeros(len(gt_ca_atoms))
    for i in range(len(gt_ca_atoms)):
        res = compute_drmsd(pred_ca_atoms[i], gt_ca_atoms[i])
        if res > 0: drmsd[i] = res
    return drmsd.mean()


def eval_drmsd_loss_batch2(x, y):
    gt_ca_atoms = x
    pred_ca_atoms = y
    drmsd = np.zeros(len(gt_ca_atoms))
    for i in range(len(gt_ca_atoms)):
        res = compute_drmsd(pred_ca_atoms[i], gt_ca_atoms[i])
        if res > 0: drmsd[i] = res
    return drmsd.mean()


def eval_drmsd_loss_reference(x, y):
    # index = torch.tensor([1], dtype=torch.long, device=batch['aatype'].device)
    # gt_atoms = batch["all_atom_positions"]
    # gt_ca_atoms = torch.squeeze(torch.index_select(gt_atoms,dim=2,index=index), dim=2)
    # pred_atoms = outputs['sm']["positions"][-1]
    # pred_ca_atoms = torch.squeeze(torch.index_select(pred_atoms,dim=2,index=index), dim=2)

    gt_ca_atoms = y["all_atom_positions"][..., :, 1, :]
    pred_ca_atoms = x['all_atom_positions'][..., :, 1, :]
    drmsd = np.zeros(len(gt_ca_atoms))
    for i in range(len(gt_ca_atoms)):
        res = compute_drmsd(pred_ca_atoms[i], gt_ca_atoms[i])
        if res > 0 : drmsd[i] = res
    return drmsd.mean()

