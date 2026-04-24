import time
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_error, mean_absolute_error
from einops import repeat

import utils
from downstream.trainer import *

class Destination(Trainer):
    def __init__(self, pre_length, **kwargs):
        super().__init__(task_name='destination', metric_type='classification', **kwargs)
        self.pre_length = pre_length
        self.loss_func = nn.CrossEntropyLoss()

    def forward_encoders(self, *x, **kwargs):
        if kwargs.get('lang', 'zh') == 'zh':
            suffix_prompt="目的地所在路段为"
        else:
            suffix_prompt="The destination is"

        if len(x) < 2:
            return super().forward_encoders(*x, suffix_prompt=suffix_prompt, d_mask=True, **kwargs)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len-self.pre_length, *x[2:], suffix_prompt=suffix_prompt, d_mask=True,
                                        **kwargs)

    def parse_label(self, label_meta):
        return label_meta.long().detach()


class TTE(Trainer):
    def __init__(self, **kwargs):
        super().__init__(task_name=f'tte', metric_type='regression', **kwargs)
        # Huber Loss (SmoothL1) prevents gradient explosion from outliers
        self.loss_func = nn.SmoothL1Loss(beta=10.0)

    def forward_encoders(self, *x, **kwargs):
        if kwargs.get('lang', 'zh') == 'zh':
            suffix_prompt="旅行时间为"
        else:
            suffix_prompt="The total travel time is"

        if len(x) < 2:
            return super().forward_encoders(*x, suffix_prompt=suffix_prompt)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len, *x[2:], suffix_prompt=suffix_prompt)

    def parse_label(self, label_meta):
        return label_meta.float()
        
    def metric_and_save(self, labels, pres, save_name):
        """ Override metric calculation to handle seconds properly """
        # Assuming inputs are already in seconds or appropriate unit
        y_true = labels.flatten()
        y_pred = pres.flatten()
        
        # Note: Removed * 60 multiplier as data.py indicates seconds are used.
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        
        metric = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
        print(metric)

        create_if_noexists(self.log_save_dir)
        metric.to_hdf(f'{self.log_save_dir}/{save_name}_{self.log_name_key}.h5',
                      key='metric', format='table')


class Search(Trainer):
    def __init__(self, **kwargs):
        super(Search, self).__init__(task_name=f'search', metric_type='classification', **kwargs)

        if 'trip_dataloader' in kwargs:
            self.trip_dataloader = kwargs['trip_dataloader']
        else:
            raise ValueError("trip_dataloader is required.")
        
        if 'neg_indices' in kwargs:
            self.neg_indices = kwargs['neg_indices'].astype(int)
        else:
            raise ValueError("neg_indices is required.")

    def train(self):
        print("Similar Trajectory Search do not require training.")
        return self.models, self.predictor

    def parse_label(self, length):
        qry_idx = list(range(int(length / 2)))
        tgt_idx = list(range(int(length / 2), length))
        return qry_idx, tgt_idx

    def eval(self, set_index, full_metric=True):
        set_name = SET_NAMES[set_index][1]
        self.eval_state()

        qrytgt_embeds = []
        for batch_meta in tqdm(self.eval_dataloader,
                               desc=f"Query/Target embeds {set_name}",
                               total=len(self.eval_dataloader), leave=False):
            batch_meta = [e.to(self.device) if isinstance(e, torch.Tensor) else e for e in batch_meta]
            encodes = self.forward_encoders(*batch_meta)
            qrytgt_embeds.append(encodes.detach().cpu().numpy())
        qrytgt_embeds = np.concatenate(qrytgt_embeds, 0)
        qry_indices, tgt_indices = self.parse_label(len(qrytgt_embeds))

        embeds = []
        for batch_meta in tqdm(self.trip_dataloader,
                               desc=f"Embeds {set_name}",
                               total=len(self.trip_dataloader), leave=False):
            batch_meta = [e.to(self.device) if isinstance(e, torch.Tensor) else e for e in batch_meta]
            encodes = self.forward_encoders(*batch_meta)
            embeds.append(encodes.detach().cpu().numpy())
        embeds = np.concatenate(embeds, 0)

        pres, labels = self.cal_pres_and_labels(qrytgt_embeds[qry_indices], qrytgt_embeds[tgt_indices], embeds[self.neg_indices])

        metric, metric_disp = 0, 0
        if self.metric_type == 'classification':
            acc = accuracy_score(labels, pres.argmax(-1))
            metric, metric_disp = acc, acc
        
        if self.use_nni:
            nni.report_final_result(metric_disp)

        if full_metric:
            self.metric_and_save(labels, pres, set_name)
        else:
            return metric, metric_disp

    def cal_pres_and_labels(self, query, target, negs):
        num_queries = query.shape[0]
        num_targets = target.shape[0]
        num_negs = negs.shape[1]
        
        query_t = repeat(query, 'nq d -> nq nt d', nt=num_targets)
        query_n = repeat(query, 'nq d -> nq nn d', nn=num_negs)
        target = repeat(target, 'nt d -> nq nt d', nq=num_queries)

        dist_mat_qt = np.linalg.norm(query_t - target, ord=2, axis=2)
        dist_mat_qn = np.linalg.norm(query_n - negs, ord=2, axis=2)
        dist_mat = np.concatenate([dist_mat_qt[np.eye(num_queries).astype(bool)][:, None], dist_mat_qn], axis=1)

        pres = -1 * dist_mat
        labels = np.zeros(num_queries)
        return pres, labels

class Classification(Trainer):
    def __init__(self, **kwargs):
        super().__init__(task_name='classification', metric_type='classification', **kwargs)
        self.loss_func = nn.CrossEntropyLoss()

    def parse_label(self, label_meta):
        return label_meta.long()

    def forward_encoders(self, *x, **kwargs):
        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len, *x[2:], suffix_prompt="该轨迹所属的司机可以被归纳为", **kwargs)
