import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from utils.math_utils import bezier_derivative,bezier_curve

class TrajPredictionEvaluator():
    ''' Return evaluation results for batched data '''

    def __init__(self, config):
        super(TrajPredictionEvaluator, self).__init__()
        self.config = config

    def evaluate(self, post_out, data):
        batch_size = len(data['PAD_FUT'])
        traj_pred = post_out['traj_pred'] # all_v,6,50,2
        prob_pred = post_out['prob_pred'] # all_v,6
        param_pred = post_out['param_pred'] # all_v,6,6,2
        # traj_pred:    batch x n_mod x pred_len x 2
        # prob_pred:    batch x n_mod

        if self.config['data_ver'] == 'av1':
            # for av1
            # traj_fut = torch.stack([traj[0, :, 0:2] for traj in data['TRAJS_FUT']])  # batch x fut x 2
            traj_fut = []
            for i in range(batch_size):
                flag = data['PAD_FUT'][i].sum(-1) >= 50 # b,k,50 -> k,50 -> k
                valid_traj = data['TRAJS_FUT'][i][flag] # k,50,2 -> v,50,2
                traj_fut.append(valid_traj)
            traj_fut = torch.cat(traj_fut, dim=0).cuda() # all_v,50,2


        elif self.config['data_ver'] == 'av2':
            # for av2
            traj_fut = torch.stack([x['TRAJS_POS_FUT'][0] for x in data["TRAJS"]])  # batch x fut x 2
        else:
            assert False, 'Unknown data_ver: {}'.format(self.config['data_ver'])


        t_values = torch.linspace(0,1,50).cuda()
        T = traj_pred.shape[2]
        traj_topk_probs, traj_topk_indices = prob_pred.topk(k=3, dim=-1)# all_v,6 -> all_v,3
        topk_trajs = traj_pred.gather(1, traj_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T,2))# all_v,6,50,2 -> all_v,3,50,2
        topk_param = param_pred.gather(1, traj_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,6, 2))# # all_v,6,6,2 -> all_v,3,6,2
        
        traj_top1_probs, traj_top1_indices = prob_pred.topk(k=1, dim=-1)# all_v,1
        top1_trajs = traj_pred.gather(1, traj_top1_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T,2))# all_v,6,50,2 -> all_v,1,50,2
        top1_param = param_pred.gather(1, traj_top1_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,6, 2))# # all_v,6,6,2 -> all_v,1,6,2

        # # jerk
        mink_RMS_jerk = get_minK_jerk(topk_param,t_values)
        min1_RMS_jerk = get_minK_jerk(top1_param,t_values)
        # # ahe
        # # fhe
        mink_ahe = get_minK_ahe(topk_trajs, traj_fut.unsqueeze(1)) # S,K,50,2    S,1,50,2    -> 1
        min1_ahe = get_minK_ahe(top1_trajs, traj_fut.unsqueeze(1))
        mink_fhe = get_minK_fhe(topk_trajs, traj_fut.unsqueeze(1))
        min1_fhe = get_minK_fhe(top1_trajs, traj_fut.unsqueeze(1))


        # to np.ndarray
        traj_pred = np.asarray(traj_pred.cpu().detach().numpy()[:, :, :, :2], np.float32) # all_v,6,50,2
        prob_pred = np.asarray(prob_pred.cpu().detach().numpy(), np.float32) # all_v,6
        param_pred = np.asarray(param_pred.cpu().detach().numpy(), np.float32) # all_v,6,6,2
        traj_fut = np.asarray(traj_fut.cpu().detach().numpy(), np.float32)# all_v,50,2

        seq_id_batch = data['SEQ_ID']
        # batch_size = len(seq_id_batch)
        all_v = traj_pred.shape[0]

        pred_dict = {}
        gt_dict = {}
        prob_dict = {}
        for idx in range(all_v):
            # seq_id = seq_id_batch[j]
            pred_dict[idx] = traj_pred[idx] # key:id, val: 6,50,2
            gt_dict[idx] = traj_fut[idx] # key:id, val: 50,2
            prob_dict[idx] = prob_pred[idx] # key:1d, val: n

        # # Max #guesses (K): 1
        res_1 = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 1, self.config['g_pred_len'], miss_threshold=self.config['miss_thres'], forecasted_probabilities=prob_dict)
        # # Max #guesses (K): 6
        res_k = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 6, self.config['g_pred_len'], miss_threshold=self.config['miss_thres'], forecasted_probabilities=prob_dict)






        eval_out = {}
        eval_out['minade_1'] = res_1['minADE']
        eval_out['minfde_1'] = res_1['minFDE']
        eval_out['mr_1'] = res_1['MR']
        eval_out['brier_fde_1'] = res_1['brier-minFDE']

        eval_out['minade_k'] = res_k['minADE']
        eval_out['minfde_k'] = res_k['minFDE']
        eval_out['mr_k'] = res_k['MR']
        eval_out['brier_fde_k'] = res_k['brier-minFDE']

        eval_out['mink_RMS_jerk'] = mink_RMS_jerk
        eval_out['min1_RMS_jerk'] = min1_RMS_jerk
        eval_out['mink_ahe'] = mink_ahe
        eval_out['min1_ahe'] = min1_ahe
        eval_out['mink_fhe'] = mink_fhe
        eval_out['min1_fhe'] = min1_fhe

        return eval_out
    

def get_minK_ahe(proposed_traj,gt_traj):
    '''
    input:
        - proposed_traj B,W不定,50,2
        - gt_traj B,1,50,2
    return 
        - ahe B,W,49 - B,1,49  -> B,W,49 -> B,W -> B -> sum
    '''  
    traj_yaw = get_yaw(proposed_traj) # B,W,49
    gt_yaw = get_yaw(gt_traj) # B,1,49
    yaw_diff = torch.abs(principal_value(traj_yaw - gt_yaw)).mean(-1).min(dim=-1).values.mean(-1) 
    return yaw_diff

def get_minK_fhe(proposed_traj, gt_traj):
    '''
    input:
        - proposed_traj B,W不定,50,2
        - gt_traj B,1,50,2
    return:
        - afe:B    B,W,49 - B,1,49  -> B,W,49, -> B,W -> B -> sum

    '''  
    traj_yaw = get_yaw(proposed_traj) # B,W,49
    gt_yaw = get_yaw(gt_traj) # B,1,49
    yaw_diff = torch.abs(principal_value(traj_yaw - gt_yaw))[...,-1].min(dim=-1).values.mean(-1) 
    return yaw_diff

def get_minK_jerk(param, t_values):
    '''
    param: [B,W,n+1, 2]
    '''
    plan_vec_param = bezier_derivative(param) # B,W, n_order,2
    plan_acc_param = bezier_derivative(plan_vec_param) # B, W,n_order-1, 2
    plan_jerk_param = bezier_derivative(plan_acc_param) # B,W, n_order-2, 2
    plan_jerk_vector = bezier_curve(plan_jerk_param,t_values)/(5**3) # B,W, 20,2
    plan_jerk_scaler = torch.linalg.norm(plan_jerk_vector,dim=-1) # B,W, 20
    plan_RMS_jerk = torch.sqrt(torch.mean(plan_jerk_scaler**2, dim=-1)).min(dim=-1).values # B,W,20 -均方根-> B,W->B
    plan_RMS_jerk = torch.mean(plan_RMS_jerk, dim=-1)# B->1
    return plan_RMS_jerk

def get_yaw(traj):
    '''
    traj:B,N,T,2 B个agent N条轨迹
    '''
    vec_vector = torch.diff(traj, dim=-2) #  B,N,T-1,2
    yaw = torch.atan2(vec_vector[...,1],vec_vector[...,0]) # B,N,T-1
    return yaw

def principal_value(angle, min_= -math.pi):
    """
    Wrap heading angle in to specified domain (multiples of 2 pi alias),
    ensuring that the angle is between min_ and min_ + 2 pi. This function raises an error if the angle is infinite
    :param angle: rad
    :param min_: minimum domain for angle (rad)
    :return angle wrapped to [min_, min_ + 2 pi).
    S,N,49
    """
    assert torch.all(torch.isfinite(angle)), "angle is not finite"

    lhs = (angle - min_) % (2 * math.pi) + min_

    return lhs
