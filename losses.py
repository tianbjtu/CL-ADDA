import numpy as np
import torch
import torch.nn.functional as F
def compute_contrastive_loss(out_1, out_2, temperature = 0.1):
    """
    compute the contrastive loss of two views of the same sample
    Args:
        out1: view 1, shape: (B, dim)
        out2: view 2, shapeL (B, dim)

    Returns:

    """
    # [2*B, D]
    batch_size = out_1.shape[0]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss