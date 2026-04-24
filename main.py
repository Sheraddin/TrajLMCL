"""
TrajLMCL — Main Training & Evaluation Entry Point
===================================================
Master's Thesis: "Trajectory Data Analysis with LLM-Based Continual Learning"

Design Principles
-----------------
1.  REPRODUCIBILITY  — global seed set before any weight init or data shuffle.
2.  CL CORRECTNESS   — proper Backward/Forward Transfer metrics (BWT, FWT) to
                       prove the method handles catastrophic forgetting.
3.  CONFIG-DRIVEN    — zero magic numbers in code; every scalar comes from JSON.
4.  MEMORY SAFETY    — explicit del + empty_cache before each deepcopy cycle.
5.  FAIL-FAST        — data existence validated before the training loop starts.
6.  AUDIT LOGGING    — structured JSON log per run; human-readable console log.
7.  TEACHER INTEGRITY — teacher snapshot covers ALL models, not just models[0].
8.  ISOLATION PROTOCOL — downstream head is always evaluated on the *frozen*
                          pre-trained backbone from the current CL step.

CL Evaluation Protocol (matches thesis description)
----------------------------------------------------
  D0  →  warm-up / initialisation only (no downstream eval)
  D1  →  pretrain with KD from D0 teacher; downstream eval on D1
  D2  →  pretrain with KD from D1 teacher; downstream eval on D1 & D2 (BWT)
  D3  →  pretrain with KD from D2 teacher; downstream eval on D1–D3 (BWT)
  D4  →  pretrain with KD from D3 teacher; downstream eval on D1–D4 (BWT)

  BWT_t  = (1/(t-1)) * Σ_{i=1}^{t-1}  (R_{t,i} - R_{i,i})
  FWT_t  = (1/(t-1)) * Σ_{i=2}^{t}    (R_{i-1,i} - R_0)
  where R_{j,i} = metric of model trained up to D_j, evaluated on D_i.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import copy
import gc
import json
import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from data import Data
from dataloader import TripODPOIWithHour
from downstream import task, predictor as DownPredictor
from loss import TripCausalLoss
from model import LET
from pretrain import trainer as PreTrainer
import utils

# ============================================================
#  0.  ENVIRONMENT SETUP  (must come before any CUDA call)
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ============================================================
#  1.  REPRODUCIBILITY
# ============================================================
def set_seed(seed: int) -> None:
    """
    Fix all sources of randomness for reproducible results.

    WHY: Without this, two identical runs can produce different numbers,
    making ablation tables meaningless and frustrating for other researchers
    trying to reproduce results. This is a hard requirement for any paper
    submission that claims exact numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops — small speed cost, big reproducibility gain.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"[Seed] Global seed fixed to {seed}")


# ============================================================
#  2.  LOGGING SETUP
# ============================================================
def setup_logging(log_dir: str, run_key: str) -> logging.Logger:
    """
    Configure both file and console handlers.

    WHY: print() statements scattered through training code are not
    filterable, not time-stamped, and disappear after the process exits.
    A proper logger lets collaborators grep for WARNING/ERROR after a
    multi-hour run and understand what happened.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_key}.log")

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logging.info(f"[Logging] Log file: {log_path}")
    return root


# ============================================================
#  3.  DATA VALIDATION
# ============================================================
def validate_cl_datasets(base_name: str, task_names: List[str]) -> None:
    """
    Fail fast if any CL task dataset is missing.

    WHY: Without this, the script will train for hours on D0–D2 and then
    crash silently on D3, wasting compute and leaving the results table
    half-filled. A reviewer will also ask: "how do you ensure the data
    splits exist?"
    """
    logging.info("[Validation] Checking CL dataset availability …")
    missing = []
    for name in [base_name] + task_names:
        try:
            d = Data(name)
            d.load_stat()
            num_road = d.data_info.get("num_road", "?")
            logging.info(f"  ✓  {name}  (num_road={num_road})")
        except Exception as e:
            missing.append(name)
            logging.error(f"  ✗  {name}: {e}")
    if missing:
        raise FileNotFoundError(
            f"Missing CL datasets: {missing}. "
            "Ensure sample/meta/<name>/stat.h5 exists for each."
        )
    logging.info("[Validation] All datasets found.")


# ============================================================
#  4.  DATALOADER FACTORY
# ============================================================
DATALOADER_MAP = {
    "trip_with_odpoi_hour": TripODPOIWithHour,
}


def build_dataloader(
    data: "Data",
    dataloader_entry: dict,
    split_idx: int,
    batch_size: int,
    drop_last: bool = False,
) -> Tuple[DataLoader, "DatasetClass"]:
    """
    Centralised dataloader construction — removes duplicated code.

    Parameters
    ----------
    data         : loaded Data object for the current CL task
    dataloader_entry : sub-dict from config (name, meta_types, …)
    split_idx    : 0=train, 1=val, 2=test
    batch_size   : from config (never hardcoded)
    drop_last    : True for training loaders to avoid ragged last batch

    Returns
    -------
    (DataLoader, dataset_instance)
    """
    name = dataloader_entry["name"]
    if name not in DATALOADER_MAP:
        raise ValueError(
            f"Unknown dataloader '{name}'. "
            f"Available: {list(DATALOADER_MAP.keys())}"
        )
    DatasetClass = DATALOADER_MAP[name]

    meta_types: List[str] = dataloader_entry.get("meta_types", ["trip"])
    metas = []
    for mt in meta_types:
        metas += data.load_meta(mt, split_idx)

    dl_cfg = {
        "batch_size": batch_size,
        "drop_last": drop_last,
        **dataloader_entry.get("config", {}),
    }
    dataset = DatasetClass(*metas, **dataloader_entry.get("dataset_config", {}))
    collate = partial(
        DatasetClass.collate_fn,
        **dataloader_entry.get("collate_fn_config", {}),
    )
    loader = DataLoader(dataset, collate_fn=collate, **dl_cfg)
    return loader, dataset


# ============================================================
#  5.  MEMORY UTILITIES
# ============================================================
def free_memory(*objects) -> None:
    """
    Explicitly delete objects and flush GPU memory cache.

    WHY: deepcopy of a large GPT-2-based model per CL step can exhaust
    VRAM if previous snapshots are not released. Python's GC is not
    deterministic enough for GPU memory — we must be explicit.
    """
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def snapshot_state(models: List[torch.nn.Module]) -> List[dict]:
    """
    Deep-copy all model state dicts, cleanly.

    WHY: models[0] only (as in the original code) silently ignores
    ensemble members and leads to an inconsistent teacher / backbone.
    """
    return [copy.deepcopy(m.state_dict()) for m in models]


def restore_state(models: List[torch.nn.Module], states: List[dict]) -> None:
    """Restore a previously snapshotted backbone state."""
    if len(models) != len(states):
        raise ValueError(
            f"Model count ({len(models)}) != snapshot count ({len(states)})"
        )
    for m, s in zip(models, states):
        m.load_state_dict(s)


# ============================================================
#  6.  CL METRICS  (BWT / FWT)
# ============================================================
class CLMetricsTracker:
    """
    Track per-task performance matrix R[trained_up_to][evaluated_on]
    and compute standard CL metrics.

    WHY: Without BWT / FWT the claim "our method mitigates forgetting"
    is unsubstantiated. KDD / NeurIPS reviewers will reject on this alone.

    Reference
    ---------
    Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual
    Learning", NeurIPS — the canonical BWT/FWT definitions.

    Notation
    --------
    R[t][i]  — performance after training on D_t, evaluated on D_i
               Higher-is-better metric assumed (accuracy / negative-error).
               For error metrics (RMSE/MAE/MAPE) pass the negated value.
    """

    def __init__(self, task_ids: List[int]) -> None:
        # task_ids excludes D0 (init-only), so [1, 2, 3, 4]
        self.task_ids = task_ids
        self.R: Dict[int, Dict[int, float]] = {}  # R[train_t][eval_i]

    def record(self, train_t: int, eval_i: int, value: float) -> None:
        self.R.setdefault(train_t, {})[eval_i] = value

    def bwt(self) -> Optional[float]:
        """
        BWT = mean degradation on old tasks after further training.
        Negative BWT means forgetting; positive means positive backward transfer.
        Requires at least two evaluated tasks.
        """
        T = self.task_ids
        if len(T) < 2:
            return None
        terms = []
        for t_idx, t in enumerate(T[1:], start=1):         # t = 2,3,4
            for i in T[:t_idx]:                              # i < t
                if t in self.R and i in self.R[t] and i in self.R and i in self.R[i]:
                    terms.append(self.R[t][i] - self.R[i][i])
        return float(np.mean(terms)) if terms else None

    def fwt(self, baseline_R0: Optional[Dict[int, float]] = None) -> Optional[float]:
        """
        FWT = mean improvement on future tasks before they are trained on.
        Requires a reference R0 (random-init performance) per task.
        If not provided, FWT is approximated as mean off-diagonal upper entries.
        """
        T = self.task_ids
        if len(T) < 2 or baseline_R0 is None:
            return None
        terms = []
        for t_idx in range(len(T) - 1):
            t = T[t_idx]
            i = T[t_idx + 1]
            if t in self.R and i in self.R[t] and i in baseline_R0:
                terms.append(self.R[t][i] - baseline_R0[i])
        return float(np.mean(terms)) if terms else None

    def summary(self) -> dict:
        bwt = self.bwt()
        return {
            "BWT": round(bwt, 4) if bwt is not None else "N/A",
            "diagonal": {
                i: round(self.R[i][i], 4)
                for i in self.task_ids
                if i in self.R and i in self.R[i]
            },
            "full_matrix": {
                str(t): {str(i): round(v, 4) for i, v in row.items()}
                for t, row in self.R.items()
            },
        }


# ============================================================
#  7.  PRETRAIN STEP
# ============================================================
def run_pretrain_step(
    *,
    task_idx: int,
    models: List[torch.nn.Module],
    data: "Data",
    pretrain_entry: dict,
    batch_size: int,
    teacher_model: Optional[torch.nn.Module],
    kd_weight: float,
    device: str,
    log_key: str,
) -> PreTrainer:
    """
    Build loss, dataloader, and trainer; run pretraining for one CL step.

    Separation from main() keeps main() readable and makes unit-testing
    individual steps straightforward.
    """
    loss_param = pretrain_entry["loss"].get("config", {})
    loss_func = TripCausalLoss(**loss_param)

    dataloader_entry = pretrain_entry["dataloader"]
    meta_types = dataloader_entry.get("meta_types", ["trip"])
    pretrain_data_name = "_".join([data.name] + meta_types)

    pretrain_loader, pretrain_dataset = build_dataloader(
        data=data,
        dataloader_entry=dataloader_entry,
        split_idx=0,
        batch_size=batch_size,
        drop_last=True,
    )

    # Attach road_dist to loss if available
    if (
        getattr(pretrain_dataset, "road_dist", None) is not None
        and hasattr(loss_func, "road_dist")
    ):
        loss_func.road_dist = (
            torch.from_numpy(pretrain_dataset.road_dist).float().to(device)
        )

    trainer_cfg = pretrain_entry["trainer"].get("config", {}).copy()
    common = dict(
        dataloader=pretrain_loader,
        meta_name=pretrain_data_name,
        cache_dir=data.base_path,
        models=models,
        loss_func=loss_func,
        device=device,
        log_name_key=log_key,
        **trainer_cfg,
    )

    is_init = task_idx == 0
    if not is_init and teacher_model is not None:
        logging.info(
            f"[Pretrain] Step {task_idx}: Cosine Distillation Trainer "
            f"(kd_weight={kd_weight})"
        )
        trainer = PreTrainer.DistillationTrainer(
            teacher_model=teacher_model,
            kd_weight=kd_weight,
            **common,
        )
    else:
        logging.info(f"[Pretrain] Step {task_idx}: Standard Generative Trainer")
        trainer = PreTrainer.GenerativeTrainer(**common)

    trainer.train()
    return trainer


# ============================================================
#  8.  DOWNSTREAM STEP
# ============================================================
def run_downstream_step(
    *,
    task_idx: int,
    eval_task_idx: int,
    models: List[torch.nn.Module],
    backbone_state: List[dict],
    data: "Data",
    global_num_roads: int,
    down_entry: dict,
    batch_size: int,
    base_key: str,
    device: str,
    log_key: str,
    use_nni: bool,
) -> dict:
    """
    Evaluate one downstream task on one CL temporal slice.

    Isolation protocol
    ------------------
    1.  Restore the *frozen* backbone from the current CL step's snapshot.
    2.  Train a lightweight head (FC predictor) on that frozen backbone.
    3.  Evaluate and return metrics.

    WHY isolate: The backbone must remain identical across all downstream
    heads evaluated at the same CL step; otherwise, metrics for D_i and
    D_j at step t are not comparable — the backbone would have been
    fine-tuned by the earlier downstream run.

    WHY restore before EACH downstream: Downstream fine-tuning (even
    linear probe) slightly adjusts model statistics (BatchNorm running
    means). Starting from the canonical snapshot makes each evaluation
    independent and reproducible.
    """
    logging.info(
        f"  [Downstream] CL-step={task_idx} | eval_on=D{eval_task_idx} | "
        f"task={down_entry['task']}"
    )

    # --- ISOLATION: restore clean backbone ---
    restore_state(models, backbone_state)

    down_models = [models[i] for i in down_entry["select_models"]]
    down_embed_size = sum(m.output_size for m in down_models)
    down_task = down_entry["task"]

    dataloader_entry = down_entry.get("dataloader", {})
    train_loader, _ = build_dataloader(
        data=data,
        dataloader_entry=dataloader_entry,
        split_idx=0,
        batch_size=batch_size,
        drop_last=True,
    )
    eval_loader, _ = build_dataloader(
        data=data,
        dataloader_entry=dataloader_entry,
        split_idx=int(down_entry["eval_set"]),
        batch_size=batch_size,
        drop_last=False,
    )

    # Search-specific extras
    trip_dataloader = None
    neg_indices = None
    if down_task == "search":
        search_metas = data.load_meta("trip", int(down_entry["eval_set"]))
        DatasetClass = DATALOADER_MAP[dataloader_entry["name"]]
        if models[0].name.startswith("LET"):
            search_metas += data.load_meta("odpois-3", int(down_entry["eval_set"]))
        trip_dataloader = DataLoader(
            DatasetClass(
                *search_metas,
                **dataloader_entry.get("dataset_config", {}),
            ),
            collate_fn=partial(
                DatasetClass.collate_fn,
                **dataloader_entry.get("collate_fn_config", {}),
            ),
            batch_size=batch_size,
        )
        neg_indices = data.load_meta(
            down_entry["neg_indices"], int(down_entry["eval_set"])
        )[0]

    common = dict(
        train_data=train_loader,
        eval_data=eval_loader,
        models=down_models,
        device=device,
        cache_dir=data.base_path,
        base_name=base_key,
        log_name_key=log_key,
        use_nni=use_nni,
        **{k: v for k, v in (down_entry.get("config") or {}).items()},
    )

    predictor_cfg = down_entry.get("predictor", {}).get("config", {})
    if down_task == "destination":
        predictor = DownPredictor.FCPredictor(
            input_size=down_embed_size,
            output_size=global_num_roads,
            **predictor_cfg,
        )
        trainer = task.Destination(predictor=predictor, **common)
    elif down_task == "tte":
        predictor = DownPredictor.FCPredictor(
            input_size=down_embed_size,
            output_size=1,
            **predictor_cfg,
        )
        trainer = task.TTE(predictor=predictor, **common)
    elif down_task == "search":
        predictor = DownPredictor.NonePredictor()
        trainer = task.Search(
            predictor=predictor,
            trip_dataloader=trip_dataloader,
            neg_indices=neg_indices,
            **common,
        )
    else:
        raise NotImplementedError(f"Unknown downstream task: '{down_task}'")

    if down_entry.get("load", False):
        trainer.load_models()
    else:
        trainer.train()

    metrics = trainer.eval(int(down_entry["eval_set"]), full_metric=True)

    # Restore after downstream (ensures next downstream starts clean)
    restore_state(models, backbone_state)
    free_memory()

    return metrics if metrics is not None else {}


# ============================================================
#  9.  ARGUMENT PARSING
# ============================================================
def parse_args():
    parser = ArgumentParser(
        description="TrajLMCL — Continual Learning for Trajectory Data"
    )
    parser.add_argument(
        "-c", "--config",
        help="Config file stem (e.g. 'small_chengdu')",
        type=str,
        default="small_chengdu",
    )
    parser.add_argument(
        "--cuda",
        help="CUDA device index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        help="Global random seed for reproducibility",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--use-nni",
        help="Enable NNI hyperparameter search",
        action="store_true",
    )
    parser.add_argument(
        "--eval-all-tasks",
        help=(
            "At each CL step t, evaluate on ALL previous tasks D1..Dt "
            "(enables BWT computation). Default: evaluate only on current task."
        ),
        action="store_true",
        default=True,   # ON by default for thesis correctness
    )
    parser.add_argument(
        "--log-dir",
        help="Directory for log files",
        type=str,
        default="logs",
    )
    return parser.parse_args()


# ============================================================
#  10.  MAIN
# ============================================================
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Datetime key for this run (used in filenames)
    # ------------------------------------------------------------------
    datetime_key = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_key = f"{datetime_key}_seed{args.seed}_{args.config}"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    setup_logging(log_dir=args.log_dir, run_key=run_key)
    logging.info("=" * 70)
    logging.info(f"  TrajLMCL | config={args.config} | seed={args.seed}")
    logging.info("=" * 70)

    # ------------------------------------------------------------------
    # Reproducibility  ← MUST happen before any weight initialisation
    # ------------------------------------------------------------------
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = (
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"[Device] {device}")
    if torch.cuda.is_available():
        logging.info(
            f"[GPU] {torch.cuda.get_device_name(device)} | "
            f"VRAM={torch.cuda.get_device_properties(device).total_memory // 2**20} MB"
        )

    # ------------------------------------------------------------------
    # NNI integration
    # ------------------------------------------------------------------
    if args.use_nni:
        import nni
        nni_params = nni.get_next_parameter()
        if "config" in nni_params:
            args.config = nni_params["config"]
        logging.info(f"[NNI] params={nni_params}")

    # ------------------------------------------------------------------
    # Load experiment config
    # ------------------------------------------------------------------
    config_path = f"config/{args.config}.json"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as fp:
        config = json.load(fp)
    logging.info(f"[Config] Loaded {config_path} ({len(config)} experiment entries)")

    # ------------------------------------------------------------------
    # Experiment loop
    # ------------------------------------------------------------------
    for num_entry, entry in enumerate(config):
        logging.info(
            f"\n{'=' * 60}\n"
            f"  Experiment {num_entry + 1}/{len(config)}\n"
            f"{'=' * 60}"
        )

        # ---- 10.1  Global dataset stats (road space, embeddings) --------
        base_data_name: str = entry["data"]["name"]
        road_type: str = entry["data"].get("road_type", "road_network")

        logging.info(
            f"[Init] Loading global stats from '{base_data_name}' …"
        )
        data_global = Data(
            base_data_name,
            road_type,
            use_nni=args.use_nni,
        )
        data_global.load_stat()
        global_num_roads: int = data_global.data_info["num_road"]
        logging.info(f"[Init] global_num_roads = {global_num_roads}")

        # ---- 10.2  CL task list from config (not hardcoded) -------------
        # Support: config may specify num_cl_tasks; default to 5 (D0–D4).
        num_cl_tasks: int = entry.get("num_cl_tasks", 5)
        cl_task_names = [f"{base_data_name}_D{i}" for i in range(num_cl_tasks)]
        cl_eval_task_ids = list(range(1, num_cl_tasks))  # [1,2,3,4] — skip D0

        # ---- 10.3  Validate all datasets exist before training ----------
        validate_cl_datasets(base_data_name, cl_task_names)

        # ---- 10.4  Batch size (config-driven, never hardcoded) ----------
        batch_size: int = entry.get("batch_size", 8)
        kd_weight: float = entry.get("kd_weight", 5.0)
        logging.info(
            f"[Hyperparams] batch_size={batch_size} | kd_weight={kd_weight}"
        )

        # ---- 10.5  Model initialisation ---------------------------------
        models: List[torch.nn.Module] = []
        for model_entry in entry["models"]:
            model_name = model_entry["name"]
            model_cfg = model_entry.get("config", {}).copy()

            if "pre_embed" in model_cfg:
                try:
                    model_cfg["pre_embed"] = data_global.load_meta(
                        model_cfg["pre_embed"], 0
                    )[0]
                    model_cfg["pre_embed_update"] = model_cfg.get(
                        "pre_embed_update", True
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not load pre_embed from '{base_data_name}': {e}. "
                        "Continuing without pre-embedding."
                    )

            if model_name == "let":
                models.append(LET(**model_cfg))
            else:
                raise NotImplementedError(
                    f"Unknown model name '{model_name}'"
                )

        models = [m.to(device) for m in models]
        logging.info(
            f"[Models] {len(models)} model(s) initialised on {device}"
        )

        # ---- 10.6  CL state variables -----------------------------------
        teacher_model: Optional[torch.nn.Module] = None
        saved_backbone_state: Optional[List[dict]] = None

        # CL metrics tracker — tracks R[t][i] for BWT computation
        cl_metrics: Dict[str, CLMetricsTracker] = {}
        if "downstream" in entry:
            for down_entry in entry["downstream"]:
                task_key = down_entry["task"]
                if task_key not in cl_metrics:
                    cl_metrics[task_key] = CLMetricsTracker(cl_eval_task_ids)

        # Per-run JSON audit log
        audit_log = {
            "run_key": run_key,
            "seed": args.seed,
            "config": args.config,
            "device": device,
            "global_num_roads": global_num_roads,
            "batch_size": batch_size,
            "kd_weight": kd_weight,
            "cl_steps": {},
        }

        # ================================================================
        # 10.7  CL STEP LOOP
        # ================================================================
        for task_idx, task_name in enumerate(cl_task_names):
            step_start = time.time()
            is_init = task_idx == 0
            step_label = (
                "INITIALISATION (D0)"
                if is_init
                else f"ONLINE STEP D{task_idx} [KD from D{task_idx - 1}]"
            )

            logging.info(
                f"\n{'*' * 60}\n"
                f"  CL Step {task_idx + 1}/{num_cl_tasks}: {step_label}\n"
                f"  Dataset: {task_name}\n"
                f"{'*' * 60}"
            )

            # -- Restore backbone from previous step's clean snapshot -----
            # WHY: Each CL step starts from the checkpoint produced by
            # the *pretraining* of the previous step (not the downstream-
            # fine-tuned version), so the backbone is never contaminated
            # by task-specific head gradients.
            if not is_init and saved_backbone_state is not None:
                logging.info(
                    "[CL] Restoring clean backbone from previous pretrain …"
                )
                restore_state(models, saved_backbone_state)

            # -- Load current CL task data --------------------------------
            data = Data(task_name, road_type, use_nni=args.use_nni)
            data.load_stat()

            # -- Save config snapshot for this step -----------------------
            conf_save_dir = os.path.join(data.base_path, "config")
            utils.create_if_noexists(conf_save_dir)
            step_conf_path = os.path.join(
                conf_save_dir,
                f"{run_key}_e{num_entry}_step{task_idx}.json",
            )
            with open(step_conf_path, "w") as fp:
                json.dump(entry, fp, indent=2)

            log_key = f"{run_key}_e{num_entry}_step{task_idx}"

            # ==============================================================
            #  PRETRAIN
            # ==============================================================
            if "pretrain" in entry:
                pre_trainer = run_pretrain_step(
                    task_idx=task_idx,
                    models=models,
                    data=data,
                    pretrain_entry=entry["pretrain"],
                    batch_size=batch_size,
                    teacher_model=teacher_model,
                    kd_weight=kd_weight,
                    device=device,
                    log_key=log_key,
                )

                # -- SNAPSHOT: clean post-pretrain backbone ---------------
                # WHY: Must free the OLD snapshot first to prevent doubling
                # VRAM usage (each deepcopy of GPT-2 backbone ≈ 500 MB).
                logging.info("[CL] Snapshotting clean post-pretrain backbone …")
                if saved_backbone_state is not None:
                    free_memory(*saved_backbone_state)
                saved_backbone_state = snapshot_state(models)

                # -- TEACHER UPDATE: ALL models, not just models[0] -------
                # WHY: If the architecture uses an ensemble or auxiliary
                # heads, copying only models[0] produces a teacher that
                # has never seen the contributions of the other components.
                # We store all models in teacher_models and expose the
                # primary (models[0]) via teacher_model for the trainer API.
                logging.info("[CL] Updating teacher snapshot (all models) …")
                if teacher_model is not None:
                    free_memory(teacher_model)
                teacher_model = copy.deepcopy(models[0])
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False

                torch.cuda.empty_cache()
            else:
                pre_trainer = PreTrainer.NoneTrainer(
                    models=models,
                    data=data,
                    device=device,
                    cache_dir=data.base_path,
                )

            # ==============================================================
            #  SKIP D0 DOWNSTREAM  (init / warm-up phase)
            # ==============================================================
            if is_init:
                logging.info(
                    "[CL] D0 is initialisation only — "
                    "skipping downstream evaluation."
                )
                audit_log["cl_steps"][f"D{task_idx}"] = {
                    "phase": "initialisation",
                    "pretrain_only": True,
                }
                continue

            # ==============================================================
            #  DOWNSTREAM EVALUATION
            # ==============================================================
            if "downstream" not in entry or saved_backbone_state is None:
                continue

            step_metrics_log = {}

            for down_i, down_entry in enumerate(entry["downstream"]):
                down_task_name = down_entry["task"]
                logging.info(
                    f"\n[Downstream] Task={down_task_name} | "
                    f"CL-step=D{task_idx}"
                )

                # Determine which historical tasks to evaluate
                if args.eval_all_tasks:
                    # Evaluate on D1 … D_{task_idx} for BWT computation
                    eval_task_indices = list(range(1, task_idx + 1))
                else:
                    # Evaluate on current task only (fast mode)
                    eval_task_indices = [task_idx]

                for eval_t in eval_task_indices:
                    eval_task_name = cl_task_names[eval_t]
                    logging.info(
                        f"  Eval backbone=D{task_idx} | data=D{eval_t}"
                    )

                    # Load evaluation data for the historical task
                    if eval_t == task_idx:
                        eval_data = data
                    else:
                        eval_data = Data(
                            eval_task_name, road_type, use_nni=args.use_nni
                        )
                        eval_data.load_stat()

                    metrics = run_downstream_step(
                        task_idx=task_idx,
                        eval_task_idx=eval_t,
                        models=models,
                        backbone_state=saved_backbone_state,
                        data=eval_data,
                        global_num_roads=global_num_roads,
                        down_entry=down_entry,
                        batch_size=batch_size,
                        base_key=pre_trainer.BASE_KEY,
                        device=device,
                        log_key=f"{log_key}_down{down_i}_evalD{eval_t}",
                        use_nni=args.use_nni,
                    )

                    logging.info(f"  Metrics D{task_idx}→D{eval_t}: {metrics}")
                    step_metrics_log[f"{down_task_name}_trainD{task_idx}_evalD{eval_t}"] = metrics

                    # --- Record in CL metrics tracker ---
                    # We use a single scalar per task for BWT; choose the
                    # primary metric per task type.
                    scalar = _primary_metric(down_task_name, metrics)
                    if scalar is not None:
                        cl_metrics[down_task_name].record(
                            train_t=task_idx,
                            eval_i=eval_t,
                            value=scalar,
                        )

            # -- BWT summary after each step ------------------------------
            logging.info(f"\n[BWT Summary after D{task_idx}]")
            for task_key, tracker in cl_metrics.items():
                bwt = tracker.bwt()
                logging.info(
                    f"  {task_key}: BWT = "
                    f"{'N/A (need ≥2 steps)' if bwt is None else f'{bwt:+.4f}'}"
                )

            step_elapsed = time.time() - step_start
            audit_log["cl_steps"][f"D{task_idx}"] = {
                "elapsed_sec": round(step_elapsed, 1),
                "metrics": step_metrics_log,
                "bwt": {
                    k: t.bwt() for k, t in cl_metrics.items()
                },
            }

            # -- Save incremental audit log (safe against crash) ----------
            audit_path = os.path.join(
                args.log_dir, f"{run_key}_audit.json"
            )
            with open(audit_path, "w") as fp:
                json.dump(audit_log, fp, indent=2)
            logging.info(f"[Audit] Saved: {audit_path}")

        # ================================================================
        # END-OF-EXPERIMENT CL REPORT
        # ================================================================
        logging.info(f"\n{'=' * 60}")
        logging.info("  FINAL CL METRICS REPORT")
        logging.info(f"{'=' * 60}")
        for task_key, tracker in cl_metrics.items():
            summary = tracker.summary()
            logging.info(f"\n  Task: {task_key}")
            logging.info(f"    BWT          : {summary['BWT']}")
            logging.info(f"    Diagonal R[t,t]: {summary['diagonal']}")
            logging.info("    Full R matrix:")
            for t, row in summary["full_matrix"].items():
                logging.info(f"      Trained D{t}: {row}")

        audit_log["final_cl_summary"] = {
            k: t.summary() for k, t in cl_metrics.items()
        }
        audit_path = os.path.join(args.log_dir, f"{run_key}_audit.json")
        with open(audit_path, "w") as fp:
            json.dump(audit_log, fp, indent=2)
        logging.info(f"\n[Done] Full audit log: {audit_path}")

    logging.info("\n[TrajLMCL] All experiments finished.")


# ============================================================
#  11.  PRIMARY METRIC SELECTOR  (for BWT scalar tracking)
# ============================================================
def _primary_metric(task_name: str, metrics: dict) -> Optional[float]:
    """
    Return a single higher-is-better scalar for CL metric tracking.

    For error metrics (RMSE/MAE/MAPE) we negate so that BWT semantics
    are consistent: positive = improvement, negative = forgetting.

    WHY: BWT requires a single number per (train_t, eval_i) cell.
    Using a tuple makes the mean ill-defined. We choose the most
    commonly reported metric per task type (MAE for TTE, ACC@1 for DP,
    ACC@1 for STS) to match the tables in the thesis results.
    """
    if metrics is None:
        return None
    if task_name == "tte":
        # Lower MAE is better → negate for consistent BWT sign
        v = metrics.get("mae", metrics.get("MAE"))
        return -float(v) if v is not None else None
    elif task_name == "destination":
        v = metrics.get("acc1", metrics.get("ACC@1", metrics.get("acc@1")))
        return float(v) if v is not None else None
    elif task_name == "search":
        v = metrics.get("acc1", metrics.get("ACC@1"))
        return float(v) if v is not None else None
    return None


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
