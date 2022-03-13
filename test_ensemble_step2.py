"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
import os

from args import get_test_args
from collections import OrderedDict, defaultdict
from json import dumps, load
from models import get_model
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(eval_file, input_dir, sub_path, test):
    # Evaluate
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)

    logits = defaultdict(list)
    fcount = 0
    for fname in os.listdir(input_dir):
        fcount += 1
        with open(os.path.join(input_dir, fname)) as f:
            record = load(f)
            for k, v in record.items():
                logits[k].append(v)

    log_p1, log_p2, ids = [], [], []
    max_length = 500
    for idx, value in logits.items():
        assert len(value) == fcount
        ids.append(int(idx))
        log_p1_ = []
        log_p2_ = []
        for r in value:
            p1_ = r[0]
            p2_ = r[1]
            p1_ = p1_ + [-1e30] * (max_length - len(p1_))
            p2_ = p2_ + [-1e30] * (max_length - len(p2_))
            log_p1_.append(p1_)
            log_p2_.append(p2_)
        log_p1.append(log_p1_)
        log_p2.append(log_p2_)
    log_p1 = torch.tensor(log_p1) # id, model, seq
    log_p2 = torch.tensor(log_p2)

    log_p1 = log_p1.mean(dim=1)
    log_p2 = log_p2.mean(dim=1)

    # Get F1 and EM scores
    p1, p2 = log_p1.exp(), log_p2.exp()
    starts, ends = util.discretize(p1, p2, 15, True) # max answer length, use v2

    idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                              ids,
                                              starts.tolist(),
                                              ends.tolist(),
                                              True)
    pred_dict.update(idx2pred)
    sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if not test:
        results = util.eval_dicts(gold_dict, pred_dict, True)
        results_list = [('F1', results['F1']),
                        ('EM', results['EM'])]
        results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        print(results_str)

    # Write submission file
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main("data/dev_eval.json", "ensemble/individual_dev", "ensemble/all_dev.csv", False)
    # main("data/test_eval.json", "ensemble/individual_test", "ensemble/all_test.csv", True)
