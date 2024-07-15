import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeter, AverageMeterForDict


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    faulthandler.enable()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, _, evaluator = loader.load()

    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        pin_memory=True)

    net.eval()

    with torch.no_grad():
        # * Validation
        val_start = time.time()
        val_eval_meter = AverageMeterForDict()
        for i, data in enumerate(tqdm(dl_val)):
            data_in = net.pre_process(data)
            out = net(data_in)
            _ = loss_fn(out, data)
            post_out = net.post_process(out)

            eval_out = evaluator.evaluate(post_out, data)
            val_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

        print('\nValidation set finish, cost {:.2f} secs'.format(time.time() - val_start))
        print('-- ' + val_eval_meter.get_info())

    print('\nExit...')

def test_latency():
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    faulthandler.enable()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, _, evaluator = loader.load()

    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        pin_memory=True)

    net.eval()

    with torch.no_grad():

        data = dict()
        circle_num = 300
        actor_num = 16
        map_ele_num = 65
        instance_num = actor_num + map_ele_num
        device = torch.device("cuda")
        data['ACTORS'] = torch.rand(actor_num, 3,20, device=device)  #7,3,50
        data['ACTORS'][:,2,:] = 1
        data['ACTOR_IDCS'] = [torch.arange(actor_num, device=device)] # [0,1,2,3,4,5,6,]
        data['LANES'] = torch.rand(map_ele_num,20,10, device=device) # [36,20,10]
        data['LANE_IDCS'] = [torch.arange(map_ele_num, device=device)]# [0,1,2,3,4,....]
        data['RPE'] = [{"scene":torch.rand(5,instance_num,instance_num,device=device),"scene_mask":None}] # [5,43,43]
        
        times = torch.zeros(circle_num) # 一次iter 的耗时

        for i in tqdm(range(circle_num)):
        # for i, data in enumerate(tqdm(dl_val)):
            data_in = net.pre_process(data)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            out = net(data_in)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times[i] = end_time - start_time

        print(times)
        print(times[10:].mean(-1).item())

    print('\nExit...')


if __name__ == "__main__":
    test_latency()
    # main()
