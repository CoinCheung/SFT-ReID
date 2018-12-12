#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SFT_torch(nn.Module):
    def __init__(self, sigma=0.1, *args, **kwargs):
        super(SFT_torch, self).__init__(*args, **kwargs)
        self.sigma = sigma

    def forward(self, emb_org):
        emb_org_norm = torch.norm(emb_org, 2, 1, True).clamp(min=1e-12)
        emb_org_norm = torch.div(emb_org, emb_org_norm)
        W = torch.mm(emb_org_norm, emb_org_norm.t())
        W = torch.div(W, self.sigma)
        T = F.softmax(W, 1)
        emb_sft = torch.mm(T, emb_org)
        return emb_sft


class SFT_np(object):
    def __init__(self, sigma=0.1, *args, **kwargs):
        self.sigma = sigma

    def __call__(self, emb_org):
        emb_org_norm = np.linalg.norm(emb_org, 2, 1).reshape((-1, 1)).clip(min=1e-12)
        emb_org_norm = emb_org / emb_org_norm
        W = np.matmul(emb_org_norm, emb_org_norm.T)
        W = W / float(self.sigma)

        W_exp = np.exp(W - np.max(W, axis=1))
        T = W_exp / np.sum(W_exp, axis=1).reshape((-1, 1))
        emb_sft = np.matmul(T, emb_org)
        return emb_sft


if __name__ == '__main__':
    sft_torch_op = SFT_torch(sigma=0.1)
    sft_np_op = SFT_np(sigma=0.1)
    emb = torch.randn(4,5)
    sft1 = sft_torch_op(emb)
    sft2 = sft_np_op(emb.numpy())
    print(sft1)
    print(sft2)


