"""
downstream/task.py
==================
Downstream task definitions for TrajLMCL.

All three eval() methods now return a proper dict so that:
  - main.py can record metrics in the audit log
  - CLMetricsTracker can compute BWT / FWT
  - parse_log.py can extract structured results

Key changes vs. original
-------------------------
TTE
  * metric_and_save() returns {"rmse": ..., "mae": ..., "mape": ...}
  * eval() overridden: calls parent inference loop, then returns the dict

Destination
  * eval() overridden: runs inference, computes ACC@1 / ACC@5 / Recall,
    returns {"acc1": ..., "acc5": ..., "recall": ...}

Search
  * metric_and_save() overridden: computes Mean Rank, ACC@1, ACC@5
    from the raw distance matrix and returns a full dict
  * eval() returns that dict (was returning None when full_metric=True)
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import repeat
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)
from tqdm import tqdm

import utils
from downstream.trainer import SET_NAMES, Trainer, create_if_noexists


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topk_accuracy(preds_logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Compute top-k accuracy.

    Parameters
    ----------
    preds_logits : (N, C)  raw logits or scores
    labels       : (N,)   ground-truth class indices
    k            : int

    Returns
    -------
    float in [0, 100]
    """
    topk = np.argsort(-preds_logits, axis=1)[:, :k]          # (N, k)
    correct = np.array([labels[i] in topk[i] for i in range(len(labels))])
    return float(correct.mean() * 100.0)


def _macro_recall(preds_logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Macro-averaged per-class recall, ignoring classes absent from labels.

    Returns float in [0, 100].
    """
    preds_cls = np.argmax(preds_logits, axis=1)
    present_classes = np.unique(labels)
    # Only score on classes that actually appear in the test set
    score = recall_score(
        labels, preds_cls,
        labels=present_classes,
        average="macro",
        zero_division=0,
    )
    return float(score * 100.0)


def _ranking_metrics(dist_matrix: np.ndarray):
    """
    Compute Mean Rank, ACC@1, ACC@5 for a retrieval task.

    Parameters
    ----------
    dist_matrix : (num_queries, 1 + num_negatives)
        Column 0  = distance to the TRUE target (lower = better match).
        Columns 1+ = distances to negatives.

    Returns
    -------
    mean_rank : float   (lower is better)
    acc1      : float   percentage in [0, 100]
    acc5      : float   percentage in [0, 100]

    Note
    ----
    Rank of the true target = number of negatives that are STRICTLY
    closer (smaller distance) than the true target, plus 1.
    We use strict < so ties go to the true target (optimistic convention).
    """
    true_dist = dist_matrix[:, 0:1]          # (Q, 1)
    neg_dists = dist_matrix[:, 1:]           # (Q, num_neg)

    # How many negatives are strictly closer than the true target?
    ranks = (neg_dists < true_dist).sum(axis=1) + 1   # (Q,)

    mean_rank = float(ranks.mean())
    acc1 = float((ranks == 1).mean() * 100.0)
    acc5 = float((ranks <= 5).mean() * 100.0)
    return mean_rank, acc1, acc5


# ─────────────────────────────────────────────────────────────────────────────
#  Destination Prediction
# ─────────────────────────────────────────────────────────────────────────────

class Destination(Trainer):
    """
    Multi-class road-segment classification (destination prediction).

    Predicts the final road segment of a partial trajectory.
    The last `pre_length` steps of the valid sequence are masked
    so the model cannot "see" the destination.
    """

    def __init__(self, pre_length: int, **kwargs):
        super().__init__(task_name="destination", metric_type="classification", **kwargs)
        self.pre_length = pre_length
        self.loss_func = nn.CrossEntropyLoss()

    # ── Encoding ─────────────────────────────────────────────────────────────

    def forward_encoders(self, *x, **kwargs):
        suffix_prompt = (
            "目的地所在路段为"
            if kwargs.get("lang", "zh") == "zh"
            else "The destination is"
        )
        if len(x) < 2:
            return super().forward_encoders(
                *x, suffix_prompt=suffix_prompt, d_mask=True, **kwargs
            )
        trip, valid_len = x[:2]
        return super().forward_encoders(
            trip, valid_len - self.pre_length, *x[2:],
            suffix_prompt=suffix_prompt, d_mask=True, **kwargs
        )

    def parse_label(self, label_meta):
        return label_meta.long().detach()

    # ── Evaluation ───────────────────────────────────────────────────────────

    def eval(self, set_index: int, full_metric: bool = True) -> dict:
        """
        Run destination prediction evaluation.

        Returns
        -------
        dict with keys: acc1, acc5, recall
        All values are percentages in [0, 100].
        """
        set_name = SET_NAMES[set_index][1]
        self.eval_state()           # sets models + predictor to eval mode

        all_preds:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch_meta in tqdm(
                self.eval_dataloader,
                desc=f"Destination eval [{set_name}]",
                total=len(self.eval_dataloader),
                leave=False,
            ):
                batch_meta = [
                    e.to(self.device) if isinstance(e, torch.Tensor) else e
                    for e in batch_meta
                ]
                # Last element in the batch is always the label tensor
                *inputs, label_meta = batch_meta
                labels_batch = self.parse_label(label_meta)           # (B,) long
                encode = self.forward_encoders(*inputs)                # (B, d)
                logits = self.predictor(encode)                        # (B, num_roads)

                all_preds.append(logits.detach().cpu().float().numpy())
                all_labels.append(labels_batch.cpu().numpy())

        preds  = np.concatenate(all_preds,  axis=0)   # (N, num_roads)
        labels = np.concatenate(all_labels, axis=0)   # (N,)

        acc1   = _topk_accuracy(preds, labels, k=1)
        acc5   = _topk_accuracy(preds, labels, k=5)
        recall = _macro_recall(preds, labels)

        metric = pd.Series(
            [acc1, acc5, recall],
            index=["acc@1", "acc@5", "recall"],
        )
        print(metric)

        # Persist to HDF5 for later analysis
        create_if_noexists(self.log_save_dir)
        metric.to_hdf(
            f"{self.log_save_dir}/{set_name}_{self.log_name_key}.h5",
            key="metric", format="table",
        )

        if self.use_nni:
            import nni
            nni.report_final_result(acc1)

        return {"acc1": acc1, "acc5": acc5, "recall": recall}


# ─────────────────────────────────────────────────────────────────────────────
#  Travel Time Estimation
# ─────────────────────────────────────────────────────────────────────────────

class TTE(Trainer):
    """
    Regression: predict total travel time (seconds) for a full trajectory.
    """

    def __init__(self, **kwargs):
        super().__init__(task_name="tte", metric_type="regression", **kwargs)
        # SmoothL1 (Huber) loss prevents gradient explosion from outliers.
        self.loss_func = nn.SmoothL1Loss(beta=10.0)

    # ── Encoding ─────────────────────────────────────────────────────────────

    def forward_encoders(self, *x, **kwargs):
        suffix_prompt = (
            "旅行时间为"
            if kwargs.get("lang", "zh") == "zh"
            else "The total travel time is"
        )
        if len(x) < 2:
            return super().forward_encoders(*x, suffix_prompt=suffix_prompt)
        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len, *x[2:], suffix_prompt=suffix_prompt)

    def parse_label(self, label_meta):
        return label_meta.float()

    # ── Metrics ──────────────────────────────────────────────────────────────

    def metric_and_save(self, labels: np.ndarray, pres: np.ndarray, save_name: str) -> dict:
        """
        Compute RMSE / MAE / MAPE and persist to HDF5.

        Returns
        -------
        dict with keys: rmse, mae, mape
        mape is stored as a fraction (e.g. 0.116), NOT percent,
        to keep it consistent with the raw sklearn convention.
        Multiply by 100 for display.
        """
        y_true = labels.flatten()
        y_pred = pres.flatten()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))

        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

        metric = pd.Series([rmse, mae, mape], index=["rmse", "mae", "mape"])
        print(metric)

        create_if_noexists(self.log_save_dir)
        metric.to_hdf(
            f"{self.log_save_dir}/{save_name}_{self.log_name_key}.h5",
            key="metric", format="table",
        )

        # ← FIX: return dict so eval() can propagate it to main.py
        return {"rmse": rmse, "mae": mae, "mape": mape}

    # ── Evaluation ───────────────────────────────────────────────────────────

    def eval(self, set_index: int, full_metric: bool = True) -> dict:
        """
        Run TTE evaluation.

        Returns
        -------
        dict with keys: rmse, mae, mape
        mape is a fraction (multiply by 100 for percent display).
        """
        set_name = SET_NAMES[set_index][1]
        self.eval_state()

        all_preds:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch_meta in tqdm(
                self.eval_dataloader,
                desc=f"TTE eval [{set_name}]",
                total=len(self.eval_dataloader),
                leave=False,
            ):
                batch_meta = [
                    e.to(self.device) if isinstance(e, torch.Tensor) else e
                    for e in batch_meta
                ]
                *inputs, label_meta = batch_meta
                labels_batch = self.parse_label(label_meta)       # (B,) float
                encode = self.forward_encoders(*inputs)            # (B, d)
                pred   = self.predictor(encode).squeeze(-1)        # (B,)

                all_preds.append(pred.detach().cpu().float().numpy())
                all_labels.append(labels_batch.cpu().numpy())

        preds  = np.concatenate(all_preds,  axis=0)
        labels = np.concatenate(all_labels, axis=0)

        result = self.metric_and_save(labels, preds, set_name)

        if self.use_nni:
            import nni
            nni.report_final_result(-result["mae"])   # NNI maximises; negate MAE

        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Similar Trajectory Search
# ─────────────────────────────────────────────────────────────────────────────

class Search(Trainer):
    """
    Embedding-based trajectory retrieval (no trainable head).

    Given a query trajectory, retrieve the most similar trajectory from a
    corpus.  Similarity is measured as L2 distance in embedding space.
    """

    def __init__(self, **kwargs):
        super().__init__(task_name="search", metric_type="classification", **kwargs)

        if "trip_dataloader" not in kwargs:
            raise ValueError("trip_dataloader is required for Search.")
        self.trip_dataloader = kwargs["trip_dataloader"]

        if "neg_indices" not in kwargs:
            raise ValueError("neg_indices is required for Search.")
        self.neg_indices = kwargs["neg_indices"].astype(int)

    # ── No training needed ───────────────────────────────────────────────────

    def train(self):
        print("Similar Trajectory Search does not require training.")
        return self.models, self.predictor

    # ── Label parsing ────────────────────────────────────────────────────────

    def parse_label(self, length: int):
        half = int(length / 2)
        return list(range(half)), list(range(half, length))

    # ── Metrics ──────────────────────────────────────────────────────────────

    def metric_and_save(
        self,
        labels: np.ndarray,
        pres: np.ndarray,
        save_name: str,
        dist_matrix: np.ndarray = None,
    ) -> dict:
        """
        Compute Mean Rank, ACC@1, ACC@5 from the retrieval distance matrix.

        Parameters
        ----------
        labels      : (Q,)  ground-truth indices — all zeros (true target
                             is always at position 0 in dist_matrix)
        pres        : (Q, 1+num_neg)  negative-distance scores (higher = closer)
        save_name   : str  used in the HDF5 file name
        dist_matrix : (Q, 1+num_neg)  raw L2 distances (lower = closer).
                      If provided, used directly; otherwise derived from pres.

        Returns
        -------
        dict with keys: mean_rank, acc1, acc5
        """
        # pres = -dist_matrix  →  dist_matrix = -pres
        if dist_matrix is None:
            dist_matrix = -pres       # (Q, 1+num_neg), lower = closer

        mean_rank, acc1, acc5 = _ranking_metrics(dist_matrix)

        metric = pd.Series(
            [mean_rank, acc1, acc5],
            index=["mean_rank", "acc@1", "acc@5"],
        )
        print(metric)

        create_if_noexists(self.log_save_dir)
        metric.to_hdf(
            f"{self.log_save_dir}/{save_name}_{self.log_name_key}.h5",
            key="metric", format="table",
        )

        return {"mean_rank": mean_rank, "acc1": acc1, "acc5": acc5}

    # ── Evaluation ───────────────────────────────────────────────────────────

    def eval(self, set_index: int, full_metric: bool = True) -> dict:
        """
        Embed all query-target pairs and corpus trajectories; run retrieval.

        Returns
        -------
        dict with keys: mean_rank, acc1, acc5
        """
        set_name = SET_NAMES[set_index][1]
        self.eval_state()

        # ── Step 1: embed query-target pairs ────────────────────────────────
        qrytgt_embeds: list[np.ndarray] = []
        for batch_meta in tqdm(
            self.eval_dataloader,
            desc=f"Query/Target embeds [{set_name}]",
            total=len(self.eval_dataloader),
            leave=False,
        ):
            batch_meta = [
                e.to(self.device) if isinstance(e, torch.Tensor) else e
                for e in batch_meta
            ]
            with torch.no_grad():
                encodes = self.forward_encoders(*batch_meta)
            qrytgt_embeds.append(encodes.detach().cpu().numpy())

        qrytgt_embeds = np.concatenate(qrytgt_embeds, axis=0)
        qry_indices, tgt_indices = self.parse_label(len(qrytgt_embeds))

        # ── Step 2: embed corpus trajectories ───────────────────────────────
        embeds: list[np.ndarray] = []
        for batch_meta in tqdm(
            self.trip_dataloader,
            desc=f"Corpus embeds [{set_name}]",
            total=len(self.trip_dataloader),
            leave=False,
        ):
            batch_meta = [
                e.to(self.device) if isinstance(e, torch.Tensor) else e
                for e in batch_meta
            ]
            with torch.no_grad():
                encodes = self.forward_encoders(*batch_meta)
            embeds.append(encodes.detach().cpu().numpy())

        embeds = np.concatenate(embeds, axis=0)

        # ── Step 3: build distance matrix ───────────────────────────────────
        pres, labels, dist_matrix = self._build_dist_matrix(
            query=qrytgt_embeds[qry_indices],
            target=qrytgt_embeds[tgt_indices],
            negs=embeds[self.neg_indices],
        )

        # ── Step 4: NNI early return ─────────────────────────────────────────
        if self.use_nni:
            import nni
            # Use acc@1 as the NNI optimisation target
            _, acc1_quick, _ = _ranking_metrics(-pres)
            nni.report_final_result(acc1_quick)

        # ── Step 5: full metric computation and persistence ──────────────────
        result = self.metric_and_save(labels, pres, save_name, dist_matrix=dist_matrix)

        # ← FIX: always return the metric dict (was returning None before)
        return result

    def _build_dist_matrix(
        self,
        query:  np.ndarray,    # (Q, d)
        target: np.ndarray,    # (Q, d)  — one true target per query
        negs:   np.ndarray,    # (Q, num_neg, d)
    ):
        """
        Build the (Q, 1 + num_neg) L2 distance matrix.

        Column 0  = distance to the true target.
        Columns 1+ = distances to the negatives.

        Returns
        -------
        pres         : (Q, 1+num_neg)  negative distances (for compatibility with parent)
        labels       : (Q,)            all zeros (true target at index 0)
        dist_matrix  : (Q, 1+num_neg)  raw L2 distances
        """
        num_queries = query.shape[0]
        num_targets = target.shape[0]
        num_negs    = negs.shape[1]

        query_t  = repeat(query,  "nq d -> nq nt d", nt=num_targets)
        query_n  = repeat(query,  "nq d -> nq nn d", nn=num_negs)
        target_r = repeat(target, "nt d -> nq nt d", nq=num_queries)

        # L2 distance to the paired true target (diagonal of the full QT matrix)
        dist_qt = np.linalg.norm(query_t - target_r, ord=2, axis=2)   # (Q, Q)
        dist_true = dist_qt[np.eye(num_queries, dtype=bool)][:, None]   # (Q, 1)

        # L2 distance to each negative
        dist_neg = np.linalg.norm(query_n - negs, ord=2, axis=2)      # (Q, num_neg)

        dist_matrix = np.concatenate([dist_true, dist_neg], axis=1)   # (Q, 1+num_neg)
        pres   = -dist_matrix                                          # higher = closer
        labels = np.zeros(num_queries, dtype=int)

        return pres, labels, dist_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  Classification (driver identification, kept for completeness)
# ─────────────────────────────────────────────────────────────────────────────

class Classification(Trainer):
    def __init__(self, **kwargs):
        super().__init__(task_name="classification", metric_type="classification", **kwargs)
        self.loss_func = nn.CrossEntropyLoss()

    def parse_label(self, label_meta):
        return label_meta.long()

    def forward_encoders(self, *x, **kwargs):
        trip, valid_len = x[:2]
        return super().forward_encoders(
            trip, valid_len, *x[2:],
            suffix_prompt="该轨迹所属的司机可以被归纳为",
            **kwargs,
        )
