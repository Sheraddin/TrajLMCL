import math
from time import time
from abc import abstractmethod
import copy

import nni
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F  # Added for Cosine Similarity
from torch import nn
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm, trange

from utils import create_if_noexists, cal_model_size
from data import SET_NAMES


class Trainer:
    """ Base class of the pre-training helper class. """

    def __init__(self, dataloader, meta_name, models, trainer_name,
                 loss_func, num_epoch, lr, device,
                 log_name_key, cache_dir, cache_epoches=False,
                 suffix='', **kwargs):
        self.dataloader = dataloader
        self.cache_dir = cache_dir
        self.models = [model.to(device) for model in models]
        self.trainer_name = trainer_name
        self.use_nni = bool(kwargs.get('use_nni', False))
        self.disable_tqdm = True if self.use_nni else False
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.cache_epoches = cache_epoches
        self.loss_func = loss_func.to(device)
        loss_name = loss_func.name

        # ── AMP support ──────────────────────────────────────────────────────
        # Pass use_amp=True, amp_dtype=torch.bfloat16, scaler=<GradScaler>
        # from hp_study_v3.py.  All three default to safe no-op values so
        # existing callers that don't pass them are completely unaffected.
        self.use_amp = bool(kwargs.get('use_amp', False))
        self.amp_dtype = kwargs.get('amp_dtype', torch.float32)
        # Use torch.amp.GradScaler (PyTorch 2.x API; cuda.amp version is deprecated)
        _default_scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.scaler = kwargs.get('scaler', _default_scaler)

        # ── Plateau early stopping for pretrain ───────────────────────────────
        # Pass plateau_stopper=PlateauEarlyStopper(...) from hp_study_v3.py.
        # Defaults to None (disabled) so existing callers are unaffected.
        self.plateau_stopper = kwargs.get('plateau_stopper', None)
        # ─────────────────────────────────────────────────────────────────────

        model_name = '_'.join([model.name for model in models])
        self.BASE_KEY = f'{trainer_name}_b{dataloader.batch_size}-lr{lr}{suffix}/{loss_name}/{meta_name}/{model_name}'
        self.model_cache_dir = f'{cache_dir}/model_cache/{self.BASE_KEY}'
        self.model_save_dir = f'{cache_dir}/model_save/{self.BASE_KEY}'
        self.log_save_dir = f'{cache_dir}/log/{self.BASE_KEY}'

        self.optimizer = torch.optim.Adam(self.gather_all_param(*self.models, self.loss_func), lr=lr)
        self.log_name_key = log_name_key

        for model in models + [loss_func]:
            print(model.name, 'size', cal_model_size(model), 'MB')

    def train(self, start=-1):
        train_logs = self.train_epoches(start)
        self.save_models()
        create_if_noexists(self.log_save_dir)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='pretrain_log')
        if self.use_nni:
            nni.report_final_result(float(train_logs['loss'].to_list()[-1]))

    def train_epoches(self, start=-1, desc='Pre-training'):
        self.train_state()
        if start > -1:
            self.load_models(start)
            print('Resumed training from epoch', start)

        train_logs = []
        desc_text = f'{desc}, avg loss %.4f'
        with trange(start+1, self.num_epoch, desc=desc_text % 0.0, disable=self.disable_tqdm) as tbar:
            for epoch_i in tbar:
                s_time = time()
                epoch_avg_loss = self.train_epoch(epoch_i)
                e_time = time()
                tbar.set_description(desc_text % epoch_avg_loss)
                train_logs.append([epoch_i, e_time - s_time, epoch_avg_loss])
                if self.use_nni:
                    nni.report_intermediate_result(float(epoch_avg_loss))
                if self.cache_epoches and epoch_i < self.num_epoch - 1:
                    self.save_models(epoch_i)

                # ── Plateau early stopping check ──────────────────────────────
                # Runs after every epoch when a stopper is provided.
                # Has zero effect when plateau_stopper is None (default).
                if self.plateau_stopper is not None:
                    if self.plateau_stopper.step(epoch_avg_loss, epoch_i):
                        print(f'  [Plateau ES] No improvement for '
                              f'{self.plateau_stopper.patience} epochs — '
                              f'stopping at epoch {epoch_i + 1}/{self.num_epoch}')
                        break
                # ─────────────────────────────────────────────────────────────

        train_logs = pd.DataFrame(train_logs, columns=['epoch', 'time', 'loss'])
        return train_logs

    def train_epoch(self, epoch_i=None):
        loss_log = []
        for batch_meta in tqdm(self.dataloader, desc=f'-->Traverse batches',
                               total=len(self.dataloader), leave=False,
                               disable=self.disable_tqdm):
            batch_meta = [e.to(self.device) if isinstance(e, torch.Tensor) else e
                          for e in batch_meta]
            self.optimizer.zero_grad()

            # ── AMP forward pass ─────────────────────────────────────────────
            # When use_amp=False (default), autocast is a no-op and the scaler
            # is disabled, so behaviour is exactly identical to the original.
            # Uses torch.amp.autocast (PyTorch 2.x API).
            with torch.amp.autocast('cuda', enabled=self.use_amp,
                                    dtype=self.amp_dtype):
                loss = self.forward_loss(batch_meta)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # ─────────────────────────────────────────────────────────────────

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    @abstractmethod
    def forward_loss(self, batch_meta):
        return self.loss_func(self.models, *batch_meta)

    @staticmethod
    def gather_all_param(*models):
        parameters = []
        for encoder in models:
            parameters += list(encoder.parameters())
        return parameters

    def save_models(self, epoch=None):
        for model in (*self.models, self.loss_func):
            if epoch is not None:
                create_if_noexists(self.model_cache_dir)
                save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
            else:
                create_if_noexists(self.model_save_dir)
                save_path = f'{self.model_save_dir}/{model.name}.model'
                print('Saved model', model.name)
            torch.save(model.state_dict(), save_path)

    def load_model(self, model, epoch=None):
        if epoch is not None:
            save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
        else:
            save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model', model.name)
        return model

    def load_models(self, epoch=None):
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model, epoch)
        self.loss_func = self.load_model(self.loss_func, epoch)

    def get_models(self):
        self.eval_state()
        return self.models

    def train_state(self):
        for model in self.models:
            model.train()
        self.loss_func.train()

    def eval_state(self):
        for model in self.models:
            model.eval()
        self.loss_func.eval()


class ContrastiveTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(trainer_name='contrastive', **kwargs)

    def forward_loss(self, batch_meta):
        return self.loss_func(self.models, *batch_meta)


class GenerativeTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(trainer_name='generative', **kwargs)

    def forward_loss(self, batch_meta):
        return self.loss_func(self.models, *batch_meta)


# --- UPGRADED: COSINE DISTILLATION TRAINER ---
class DistillationTrainer(GenerativeTrainer):
    """
    SOTA Trainer with Cosine Similarity Knowledge Distillation.
    Instead of forcing exact matches (MSE), we force directional alignment.
    This prevents model collapse on domain shifts (D2->D3).

    AMP and plateau early stopping are inherited automatically from the
    base Trainer class — no extra changes needed here.
    """
    def __init__(self, teacher_model=None, kd_weight=5.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight
        
        if self.teacher_model:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            print(f">>> SOTA Knowledge Distillation (Cosine) Enabled. KD Weight: {self.kd_weight}")

    def forward_loss(self, batch_meta):
        # 1. Task Loss (Standard LLM loss)
        task_loss = self.loss_func(self.models, *batch_meta)
        
        # 2. Distillation Loss (Cosine Similarity)
        distillation_loss = 0.0
        if self.teacher_model is not None:
            trip, valid_len, o_pois, d_pois, start_weekday, start_hour = batch_meta
            
            with torch.no_grad():
                # Teacher Hidden States
                # torch.no_grad() already disables grad; autocast context from
                # train_epoch is still active here, so teacher inference is
                # also accelerated under AMP when enabled.
                teacher_h, _ = self.teacher_model.forward_latent(
                    trip, valid_len, o_pois, d_pois, start_weekday, start_hour, recover_type='trip'
                )

            # Student Hidden States
            student_model = self.models[0]
            student_h, _ = student_model.forward_latent(
                trip, valid_len, o_pois, d_pois, start_weekday, start_hour, recover_type='trip'
            )
            
            # --- THE MAGIC FIX: Cosine Similarity Loss ---
            # Loss = 1 - CosineSimilarity. 
            # If vectors point in same direction, cos=1, loss=0.
            # If vectors are opposite, cos=-1, loss=2.
            # Unlike MSE, this ignores vector MAGNITUDE, allowing adaptation to new domains.
            
            # Align shapes: (Batch, Seq_Len, Hidden_Dim) -> Flatten to (Batch*Seq_Len, Hidden_Dim) for calculation
            student_flat = student_h.view(-1, student_h.size(-1))
            teacher_flat = teacher_h.view(-1, teacher_h.size(-1))
            
            # Calculate Cosine Embedding Loss
            # Target is 1.0 (we want them to be similar)
            target = torch.ones(student_flat.size(0)).to(self.device)
            distillation_loss = F.cosine_embedding_loss(student_flat, teacher_flat, target)
        
        # 3. Combine
        total_loss = task_loss + (self.kd_weight * distillation_loss)
        return total_loss

    def train_state(self):
        super().train_state()
        if self.teacher_model:
            self.teacher_model.eval()


class NoneTrainer():
    def __init__(self, models, data, cache_dir, device):
        self.models = [model.to(device) for model in models]
        self.BASE_KEY = f'end2end/none/{data.name}'
        self.device = device
        self.model_save_dir = f'{cache_dir}/model_save/{self.BASE_KEY}'

    def save_models(self):
        create_if_noexists(self.model_save_dir)
        for model in self.models:
            save_path = f'{self.model_save_dir}/{model.name}.model'
            torch.save(model.state_dict(), save_path)

    def load_model(self, model):
        save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model from', save_path)
        return model

    def load_models(self):
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model)

    def get_models(self):
        for model in self.models:
            model.eval()
        return self.models
