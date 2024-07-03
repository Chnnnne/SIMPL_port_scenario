from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long


class LossFunc(nn.Module):
    def __init__(self, config, device):
        super(LossFunc, self).__init__()
        self.config = config
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, data):
        # print('TRAJS_FUT: ', len(data["TRAJS_FUT"]), data["TRAJS_FUT"][0].shape)
        # print('PAD_FUT: ', len(data["PAD_FUT"]), data["PAD_FUT"][0].shape)
        # print('out: ', out[1][0].shape, out[0][0].shape)
        loss_out = self.pred_loss(out,
                                  gpu(data["TRAJS_FUT"], self.device),
                                  to_long(gpu(data["PAD_FUT"], self.device)))
        loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"]
        return loss_out

    def pred_loss(self, out: Dict[str, List[torch.Tensor]], gt_preds: List[torch.Tensor], pad_flags: List[torch.Tensor]):
        '''
        input:
            out = res_cls, res_reg, res_aux
            cls = [ (4,6), (2,6), (5,6)]
            reg = [(4,6,T,2), (2,6,T,2), (5,6,T,2)]
            gt_preds = [(4,50,2),(2,50,2),(5,50,2) ]
            pad_flags = [(4,50),(2,50),(5,50)]
        cal:
            cls_loss:
            reg_loss:
        '''
        cls = out[0]
        reg = out[1]
        # cls = torch.cat([x[0:2] for x in cls], 0)
        # reg = torch.cat([x[0:2] for x in reg], 0)
        # gt_preds = torch.cat([x[0:2] for x in gt_preds], 0)
        # has_preds = torch.cat([x[0:2] for x in pad_flags], 0).bool()
        cls = torch.cat([x for x in cls], 0) # [all_n, modes] 所有batch的所有agent进行聚合
        reg = torch.cat([x for x in reg], 0) # [all_n, modes, 50, 2]
        gt_preds = torch.cat([x for x in gt_preds], 0) # [all_n,50, 2]
        has_preds = torch.cat([x for x in pad_flags], 0).bool() # [all_n,50]

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = 50
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds) # all_n,50
        max_last, last_idcs = last.max(1) # all_n
        mask = max_last > 1.0 # all_n 标志该agent是否存在有效future

        cls = cls[mask] # [K, modes]
        reg = reg[mask] # [K, modes, 50, 2]
        gt_preds = gt_preds[mask] # [K, 50, 2]
        has_preds = has_preds[mask] # [K, 50]
        last_idcs = last_idcs[mask] # [K] 标记有效的最后一个idx

        _reg = reg[..., 0:2].clone()  # K, modes, 50, 2 for WTA strategy, in case of (5-dim) prob output

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)# [0,1,...,K]
        dist = []
        for j in range(num_modes):
            dist.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])** 2 # K, 2 - K,2 -> K
                    ).sum(1)
                )
            ) # list:len6[K,K,K] 每个item是mode i下的预测和真值的fde
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1) # K,6

        min_dist, min_idcs = dist.min(1) # K, K
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        # cls loss
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls # K, modes -> K,1 - K,6 -> K,6     agent的各个轨迹与“距离真值最近的轨迹”的相对cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1) # K -> K,1  每个agent的预测轨迹距离真值最小的那条如果小于2m，标记
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"] # K,6 - K, 1 -> K,6 每个agent的每条预测轨迹距离差大于0.2m的，标记
        mgn = mgn[mask0 * mask1] # K,6[K,1*K,6]->valid 过滤agent的冗余traj和 距离真值最近traj仍太远的agent

        mask = mgn < self.config["mgn"] # vlaid 距离“距离真值轨迹最近的预测轨迹”的相对cls足够小才认为是有效的
        num_cls = mask.sum().item() # 总有效traj数
        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss
        # reg loss
        reg = reg[row_idcs, min_idcs] # [K, K] -> K, 50, 2  距离真值最近的轨迹
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10) # K,50,2[K,50] -> reg_loss(V,2 + V,2)/point num  -> 1  
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss

        return loss_out

    def print(self):
        print('\nloss_fn config:', self.config)
        print('loss_fn device:', self.device)
