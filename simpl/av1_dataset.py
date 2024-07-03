import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
import pandas as pd
#
import torch
from torch.utils.data import Dataset
#
from utils.utils import from_numpy
import pickle


class ArgoDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 20,
                 pred_len: int = 30,
                 aug: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.aug = aug
        self.verbose = verbose

        self.dataset_files = []
        self.dataset_len = -1
        self.prepare_dataset(dataset_dir) # assign dataset_files

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        if self.verbose:
            print('[Dataset] Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)
            print('-- aug: ', self.aug)

    def prepare_dataset(self, feat_path):
        if self.verbose:
            print("[Dataset] preparing {}".format(feat_path))

        if isinstance(feat_path, list):
            for path in feat_path:
                sequences = os.listdir(path)
                sequences = sorted(sequences)
                for seq in sequences:
                    file_path = f"{path}/{seq}"
                    self.dataset_files.append(file_path)
        else:
            sequences = os.listdir(feat_path)
            sequences = sorted(sequences)
            for seq in sequences:
                file_path = f"{feat_path}/{seq}"
                self.dataset_files.append(file_path)

        self.dataset_len = len(self.dataset_files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        data_path = self.dataset_files[idx]
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)
        data = {}
        data['SEQ_ID'] = data_path.split('/')[-1][:-4]
        data['CITY_NAME'] = "YongZhou"
        data['ORIG'] = data_dict['agent_ctrs'][0] # 2
        cos_, sin_ = data_dict['agent_vecs'][0]
        data['ROT'] = np.array([[cos_, -sin_],[sin_, cos_]]) # 2,2
        
        data['TRAJS_OBS'] = data_dict['agent_feats'][:,:,:2] # n,20,2 此处读取的是pkl中all_agent的历史信息。由于all agent是最大值，所以此处没有关于agent的pad只有帧的pad
        # 检查数组是否包含 Inf 值
        if np.isinf(data['TRAJS_OBS']).any():
            print("The array contains Inf values            ||| \n")
        data['TRAJS_FUT'] = data_dict['gt_preds'] # n,50,2
        data['TRAJS_FUT'][0] = data_dict['ego_gt_traj'] # 50, 2
        data['PAD_OBS'] = data_dict['agent_mask'] # n, 20
        agent_num = data_dict['agent_mask'].shape[0] # n包含target和surround
        # pad_flag = np.sum(data['PAD_OBS'],axis=-1) > 0 # n
        pad_flag = data_dict['candidate_mask'].sum(-1) > 0 # N,M-> N->N
        pad_future = np.zeros((agent_num,50),dtype=int)# n,50
        pad_future[pad_flag] = 1

        data['PAD_FUT'] = pad_future # n,50
        data['TRAJS_CTRS'] = data_dict['agent_ctrs'] # n,2
        data['TRAJS_VECS'] = data_dict['agent_vecs'] # n,2


        graph = {}
        '''
            lane-seg-points
            1个lane,164个seg, 1个seg 10个point

            graph结构：
            'node_ctrs'         (164, 10, 2) 1
            'node_vecs'         (164, 10, 2) 1
            'turn'              (164, 10, 2)
            'control'           (164, 10)
            'intersect'         (164, 10)
            'left'              (164, 10)
            'right'             (164, 10)
            'lane_ctrs'         (164, 2) 1
            'lane_vecs'         (164, 2) 1
            'num_nodes'         1640
            'num_lanes'         164
            'rel_lane_flags'    (164,)
        '''
        map_n = data_dict['map_feats'].shape[0]

        graph['node_ctrs'] = data_dict['map_feats'][:,:,:2] # map_n, 20,2
        graph['node_vecs'] = data_dict['map_feats'][:,:,2:4] # map_n, 20,2
        graph['turn'] = np.zeros_like(graph['node_vecs']) # map_n, 20,2


        graph['control'] = np.zeros((map_n, 20)) # map_n, 20 
        graph['intersect'] = np.zeros((map_n, 20)) # map_n, 20
        graph['left'] = np.zeros((map_n, 20)) # map_n, 20
        graph['right'] = np.zeros((map_n, 20)) # map_n, 20

        graph['lane_ctrs'] = data_dict['map_ctrs'] # map_n, 2
        graph['lane_vecs'] = data_dict['map_vecs'] # map_n, 2

        graph['num_lanes'] = map_n
        graph['num_nodes'] = graph['num_lanes'] * 20
        graph['rel_lane_flags'] = np.zeros(map_n)

        data['LANE_GRAPH'] = graph 
        rpes= {}
        rpes['scene'], rpes['scene_mask'] = data_dict['rpe'], None
        rpes['scene'] = np.transpose(rpes['scene'], (2,0,1))
        data['RPE'] = rpes

        return data

    # def __getitem__(self, idx):
    #     df = pd.read_pickle(self.dataset_files[idx])
    #     '''
    #         "SEQ_ID", 
    #         "CITY_NAME",
    #         "ORIG", [1,2]
    #         "ROT", [2,2]
    #         "TIMESTAMP", (1,50)
    #         "TRAJS", (n,50,2)
    #         "TRAJS_CTRS", (n,2)
    #         "TRAJS_VECS", (n,2)
    #         "PAD_FLAGS",(n,50)

    #         "LANE_GRAPH":
    #             - node
    #     '''

    #     data = self.data_augmentation(df)

    #     seq_id = data['SEQ_ID']
    #     city_name = data['CITY_NAME']
    #     orig = data['ORIG']
    #     rot = data['ROT']

    #     # timestamp = data['TIMESTAMP']
    #     trajs = data['TRAJS']
    #     trajs_obs = trajs[:, :self.obs_len]
    #     trajs_fut = trajs[:, self.obs_len:]

    #     pad_flags = data['PAD_FLAGS']
    #     pad_obs = pad_flags[:, :self.obs_len]
    #     pad_fut = pad_flags[:, self.obs_len:]

    #     trajs_ctrs = data['TRAJS_CTRS'] # （num of object, 2）
    #     trajs_vecs = data['TRAJS_VECS']

    #     graph = data['LANE_GRAPH']
    #     # for k, v in graph.items():
    #     #     print(k, type(v), v.shape if type(v) == np.ndarray else [])
    #     '''
    #         lane-seg-points
    #         1个lane,164个seg, 1个seg 10个point

    #         graph结构：
    #         'node_ctrs'         (164, 10, 2) 1
    #         'node_vecs'         (164, 10, 2) 1
    #         'turn'              (164, 10, 2)
    #         'control'           (164, 10)
    #         'intersect'         (164, 10)
    #         'left'              (164, 10)
    #         'right'             (164, 10)
    #         'lane_ctrs'         (164, 2) 1
    #         'lane_vecs'         (164, 2) 1
    #         'num_nodes'         1640
    #         'num_lanes'         164
    #         'rel_lane_flags'    (164,)
    #     '''

    #     lane_ctrs = graph['lane_ctrs']
    #     lane_vecs = graph['lane_vecs']

    #     # ~ calc rpe
    #     rpes = dict()
    #     #  trajs_ctrs(每个object的obs处的坐标，以agent为坐标系, Na,2) cat with lane_ctrs(每个seg的中点, N_seg, 2)
    #     # scene_ctrs  shape(Na+N_seg,2)
    #     scene_ctrs = torch.cat([torch.from_numpy(trajs_ctrs), torch.from_numpy(lane_ctrs)], dim=0)# all,2
    #     scene_vecs = torch.cat([torch.from_numpy(trajs_vecs), torch.from_numpy(lane_vecs)], dim=0)# all,2
    #     rpes['scene'], rpes['scene_mask'] = self._get_rpe(scene_ctrs, scene_vecs)

    #     data = {}
    #     data['SEQ_ID'] = seq_id
    #     data['CITY_NAME'] = city_name
    #     data['ORIG'] = orig
    #     data['ROT'] = rot
    #     data['TRAJS_OBS'] = trajs_obs
    #     data['TRAJS_FUT'] = trajs_fut
    #     data['PAD_OBS'] = pad_obs
    #     data['PAD_FUT'] = pad_fut
    #     data['TRAJS_CTRS'] = trajs_ctrs
    #     data['TRAJS_VECS'] = trajs_vecs

    #     data['LANE_GRAPH'] = graph # 得到这些
    #     data['RPE'] = rpes

    #     return data

    def _get_cos(self, v1, v2):
        ''' input: [M, N, 2], [M, N, 2]
            output: [M, N]
            cos(<a,b>) = (a dot b) / |a||b|
        '''
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        cos_dang = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
        return cos_dang

    def _get_sin(self, v1, v2):
        ''' input: [M, N, 2], [M, N, 2]
            output: [M, N]
            sin(<a,b>) = (a x b) / |a||b|
        '''
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        sin_dang = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
        return sin_dang

    def _get_rpe(self, ctrs, vecs, radius=100.0):
        # distance encoding
        # ctrs(Na+N_seg,2)
        # return rpe shape(5, Na+N_seg, Na+N_seg)
        d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
        if False:
            mask = d_pos >= radius
            d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
            pos_rpe = []
            for l_pos in range(10):
                pos_rpe.append(torch.sin(2**l_pos * math.pi * d_pos))
                pos_rpe.append(torch.cos(2**l_pos * math.pi * d_pos))
            # print('pos rpe: ', [x.shape for x in pos_rpe])
            pos_rpe = torch.stack(pos_rpe)
            # print('pos_rpe: ', pos_rpe.shape)
        else:
            mask = None
            d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
            pos_rpe = d_pos.unsqueeze(0)
            # print('pos_rpe: ', pos_rpe.shape)

        # angle diff
        cos_a1 = self._get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
        sin_a1 = self._get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
        # print('cos_a1: ', cos_a1.shape, 'sin_a1: ', sin_a1.shape)

        v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
        cos_a2 = self._get_cos(vecs.unsqueeze(0), v_pos)
        sin_a2 = self._get_sin(vecs.unsqueeze(0), v_pos)
        # print('cos_a2: ', cos_a2.shape, 'sin_a2: ', sin_a2.shape)

        ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
        rpe = torch.cat([ang_rpe, pos_rpe], dim=0)
        return rpe, mask

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        batch = from_numpy(batch)
        data = dict()
        data['BATCH_SIZE'] = len(batch)
        # Batching by use a list for non-fixed size
        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]
        '''
            Keys:
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'PAD_OBS', 'TRAJS_FUT', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS',
            'LANE_GRAPH',
            'RPE'
        '''
        # batch中每个sample按照key拆分成一个个list放进data dict里
        # if torch.data['TRAJS_OBS']
        actors, actor_idcs = self.actor_gather(data['BATCH_SIZE'], data['TRAJS_OBS'], data['PAD_OBS'])
        lanes, lane_idcs = self.graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])

        data['ACTORS'] = actors
        data['ACTOR_IDCS'] = actor_idcs
        data['LANES'] = lanes
        data['LANE_IDCS'] = lane_idcs
        return data

    def actor_gather(self, batch_size, actors, pad_flags):
        ''' 
        input:
            - data['TRAJS_OBS'] == actors  (bs, num_agent, 20, 2) 
            - pad_flags (bs, num_agent, 20）
        output:
            - actors    (bs, Na, 3, 20)->(all_Num, 3, 20)
            - actor_idcs    [tensor([0, 1, 2, 3]), tensor([ 4,  5,  6,  7,  8,  9, 10])] 每个sample(pkl)的agent在actors对应的idx
        
        '''
        num_actors = [len(x) for x in actors]# 每个sample（pkl）的agent数量

        act_feats = []
        for i in range(batch_size):#遍历每个sample(pkl)
            # actors[i] (N_a, 20, 2)
            # pad_flags[i] (N_a, 20)
            vel = torch.zeros_like(actors[i]) # vel:(N_a, 20, 2)
            vel[:, 1:, :] = actors[i][:, 1:, :] - actors[i][:, :-1, :] # Na, 20, 2
            act_feats.append(torch.cat([vel, pad_flags[i].unsqueeze(2)], dim=2)) # pad_flags[i] (N_a, 20, 1)  -> cat (Na, 20, 3) -> list[]
        act_feats = [x.transpose(1, 2) for x in act_feats] # act_feats(bs, Na, 20, 3) - > (bs, Na, 3, 20)
        actors = torch.cat(act_feats, 0)  #   (bs, Na, 3, 20)->(all_Num of batch, 3, 20)
        if torch.isnan(actors).any():
            print("atcor")
            exit()
        actor_idcs = []  # e.g. [tensor([0, 1, 2, 3]), tensor([ 4,  5,  6,  7,  8,  9, 10])]
        count = 0
        # 由于现在的actor组成形式是(all_Num of batch, 3, 20)  3代表xy_delta(也即xy方向上的v)和pad
        # 所以要有actor_idcs
        for i in range(batch_size):
            idcs = torch.arange(count, count + num_actors[i])
            actor_idcs.append(idcs)
            count += num_actors[i]
        return actors, actor_idcs

    def graph_gather(self, batch_size, graphs):
        # graphs:list  graphs[i]是一个sample的lane_graph
        '''
            graphs[i]
                node_ctrs           torch.Size([116, 10, 2])  116是seg num, 10是每个seg等距采10个点，2是xy
                node_vecs           torch.Size([116, 10, 2])
                turn                torch.Size([116, 10, 2])
                control             torch.Size([116, 10])
                intersect           torch.Size([116, 10])
                left                torch.Size([116, 10])
                right               torch.Size([116, 10])
                lane_ctrs           torch.Size([116, 2])
                lane_vecs           torch.Size([116, 2])
                num_nodes           1160
                num_lanes           116
        '''
        lane_idcs = list()
        lane_count = 0
        for i in range(batch_size):
            l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
            lane_idcs.append(l_idcs)
            lane_count = lane_count + graphs[i]["num_lanes"]
        # print('lane_idcs: ', lane_idcs)

        graph = dict()
        for key in ["node_ctrs", "node_vecs", "turn", "control", "intersect", "left", "right"]:
            graph[key] = torch.cat([x[key] for x in graphs], 0)#将每个sample的key字段堆叠在一起 graph[key] = (stack_num, 10, 2)
            # print(key, graph[key].shape)
        for key in ["lane_ctrs", "lane_vecs"]:
            graph[key] = [x[key] for x in graphs] # list
            # print(key, [x.shape for x in graph[key]])

        lanes = torch.cat([graph['node_ctrs'],
                           graph['node_vecs'],
                           graph['turn'],
                           graph['control'].unsqueeze(2),
                           graph['intersect'].unsqueeze(2),
                           graph['left'].unsqueeze(2),
                           graph['right'].unsqueeze(2)], dim=-1)  # [N_{lane}, 10, F] F=10= 2+2+2+1+1+1+1

        return lanes, lane_idcs

    def rpe_gather(self, rpes):
        rpe = dict()
        for key in list(rpes[0].keys()):
            rpe[key] = [x[key] for x in rpes]
        return rpe

    def data_augmentation(self, df):
        '''
            "SEQ_ID", "CITY_NAME", "ORIG", "ROT",
            "TIMESTAMP", "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "PAD_FLAGS", "LANE_GRAPH"

            "node_ctrs", "node_vecs",
            "turn", "control", "intersect", "left", "right"
            "lane_ctrs", "lane_vecs"
            "num_nodes", "num_lanes", "node_idcs", "lane_idcs"
        '''

        data = {}
        for key in list(df.keys()):
            data[key] = df[key].values[0]

        is_aug = random.choices([True, False], weights=[0.3, 0.7])[0]
        if not (self.aug and is_aug):
            return data

        # ~ random vertical flip
        data['TRAJS_CTRS'][..., 1] *= -1
        data['TRAJS_VECS'][..., 1] *= -1
        data['TRAJS'][..., 1] *= -1

        data['LANE_GRAPH']['lane_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['lane_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['node_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['node_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['left'], data['LANE_GRAPH']['right'] = data['LANE_GRAPH']['right'], data['LANE_GRAPH']['left']

        return data
