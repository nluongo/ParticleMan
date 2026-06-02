#!/usr/bin/env python3
"""
Hyperparameter optimisation for ParticleMan using Optuna.

Tunes model size hyperparameters (d_model, n_heads, n_layers, dropout) with
Optuna's TPE sampler and MedianPruner. Each trial trains for a fixed number
of epochs and reports best validation loss. Results are persisted to an SQLite
study so interrupted runs can be resumed.

Usage:
    # Run HPO (creates/resumes study at hpo_study.db)
    python scripts/hpo.py

    # Override data config, epochs per trial, and number of trials
    python scripts/hpo.py --data-config bbllv08_classify --epochs 5 --trials 30

    # Use 'classify' mode (supervised) instead of default 'pretrain'
    python scripts/hpo.py --mode classify --data-config bbllv08_classify

    # Change study name or storage path
    python scripts/hpo.py --study-name my_study --storage sqlite:///my_study.db

    # Resume a previous study
    python scripts/hpo.py --study-name particleman_hpo --storage sqlite:///hpo_study.db
"""

import os
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

import argparse
import logging
from mpi4py import MPI
import optuna
import sys
import torch
import torch.distributed as dist
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from particleman.config import TrainConfig
from particleman.data import create_dataloaders, create_datasets
from particleman.models import (
    AttentionBiasTransformer,
    EmbeddingType,
    ParticleConfig,
    ParticleTransformer,
    PhiEncoding,
)
from particleman.training import (
    ParticleTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    create_distributed_dataloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data_config(data_config_name: str) -> dict:
    configs_dir = Path(__file__).parent.parent / "configs"
    path = configs_dir / "data" / f"{data_config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Data config not found: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

def get_loaders(cfg, is_distributed=False, rank=None, world_size=None):
    # Load dataloaders once — shared across all trials to avoid repeated I/O.
    logger.info("Loading data (shared across trials)...")

    max_events=cfg.data.max_events
    try:
        if is_distributed:
            # For distributed training, create datasets then wrap with distributed sampler
            train_dataset, val_dataset, _ = create_datasets(
                cfg.data,
                max_events=max_events,
            )
            
            seed = cfg.data.get("split", {}).get("seed", 42)
            
            train_loader = create_distributed_dataloader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                rank=rank,
                world_size=world_size,
                seed=seed,
            )
            
            val_loader = create_distributed_dataloader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                rank=rank,
                world_size=world_size,
                drop_last=False,
                seed=seed,
            )
            
        else:
            # Single GPU: use standard dataloaders
            train_loader, val_loader, _ = create_dataloaders(
                cfg.data,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                max_events=max_events,
            )
    except Exception as e:
        if is_main_process(rank):
            logger.error(f"Failed to create dataloaders: {e}")
            logger.error("Make sure the ROOT file exists and the config is correct.")
        cleanup_distributed()
        sys.exit(1)
    
    if is_main_process(rank):
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        if is_distributed:
            logger.info(f"Effective batch size: {cfg.training.batch_size * world_size}")

    return train_loader, val_loader

def build_objective(cfg: dict):
    """Return an Optuna objective function closed over data and CLI args."""

    data_cfg = cfg.data

    # Setup distributed environment (returns rank=0, world_size=1 for single GPU)
    rank, world_size, device = setup_distributed(backend=["gloo", "nccl"])
    is_distributed = world_size > 1
 
    train_loader, val_loader = get_loaders(cfg, is_distributed, rank, world_size)
    
    preproc = data_cfg.get("preprocessing", {})
    max_particles = preproc.get("max_particles", 200)
    eta_cut = preproc.get("eta_cut", 5.0)
    n_event_types = data_cfg.get("n_event_types", 0)

    #torch.cuda.set_device(local_rank)
    def objective(trial: optuna.Trial) -> float:
        base_trial = trial if rank == 0 else None
        print(f"Rank {rank} is passing trial={base_trial} to TorchDistributedTrial", flush=True)
        #if is_distributed:
        trial = optuna.integration.TorchDistributedTrial(base_trial)
        print("Set the trial")

        # --- Suggest model-size hyperparameters ---
        d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
        # n_heads must divide d_model evenly
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        n_heads = trial.suggest_categorical("n_heads", valid_heads)
        n_layers = trial.suggest_int("n_layers", 6, 12)
        #dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        d_model_ff_mult = trial.suggest_categorical("d_model_ff_mult", [1, 2, 4, 8])

        model_config = ParticleConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * d_model_ff_mult,
            dropout=0.1,
            max_particles=max_particles,
            n_particle_types=16,
            mask_prob=cfg.training.mask_prob,
            embedding_type=EmbeddingType.JOINT,
            phi_encoding=PhiEncoding.RAW,
            pt_range=(0.0, 500.0),
            eta_range=(-eta_cut, eta_cut),
            phi_range=(-3.14159, 3.14159),
            n_event_types=n_event_types,
            event_label_mask_prob=cfg.training.mask_prob,
        )

        n_params = sum(
            p.numel()
            for p in (
                ParticleTransformer(model_config) if not cfg.model.angular_attention_bias
                else AttentionBiasTransformer(model_config)
            ).parameters()
        )
        logger.info(
            f"Trial {trial.number}: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, — {n_params:,} params"
        )

        model = (
            AttentionBiasTransformer(model_config)
            if cfg.model.angular_attention_bias
            else ParticleTransformer(model_config)
        )

        trainer = ParticleTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=cfg.training.epochs,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            device=device,
            rank=rank,
            world_size=world_size,
            log_interval=999_999,  # suppress per-step logging
            save_dir=None,
            mode=cfg.training.mode,
        )

        best_val_loss = float("inf")
        for epoch in range(cfg.training.epochs):
            # Train one epoch manually so we can prune after each epoch.
            trainer.epoch = epoch
            trainer.optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_loader):
                losses = trainer.train_step(batch)
                if losses is None:
                    continue
                if (batch_idx + 1) % trainer.gradient_accumulation_steps == 0:
                    import torch.nn.utils as nn_utils
                    nn_utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
                    trainer.optimizer.step()
                    trainer.scheduler.step()
                    trainer.optimizer.zero_grad()
                    trainer.step += 1

            val_losses = trainer.validate()
            val_loss = val_losses.get("total_loss", float("inf"))
            best_val_loss = min(best_val_loss, val_loss)

            trial.report(val_loss, epoch)
            #if trial.should_prune():
            #    logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}")
            #    raise optuna.TrialPruned()

        logger.info(f"Trial {trial.number} finished — best val loss: {best_val_loss:.4f}")
        return best_val_loss

    return objective

#class CleanupCallback:
#    def __init__(self):
#        pass
#
#    def __call__(self, study, trial):
#        duplicative_trials = study.get_trials(deepcopy=False)
#        for trial in duplicative_trials:
#            if trial.params == {}:
#
#            print(trial)
#            print(trial.user_attrs)
#            print(trial.user_attrs.get("params_d_model", 0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: TrainConfig) -> None:
    #parser = argparse.ArgumentParser(description="ParticleMan HPO with Optuna")
    #parser.add_argument("--data-config", default="default", help="Data config name (without .yaml)")
    #parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    #parser.add_argument("--batch-size", type=int, default=256)
    #parser.add_argument("--lr", type=float, default=1e-4)
    #parser.add_argument("--weight-decay", type=float, default=0.01)
    #parser.add_argument("--mask-prob", type=float, default=0.15)
    #parser.add_argument("--mode", default="pretrain", choices=["pretrain", "classify"])
    #parser.add_argument("--max-events", type=int, default=None, help="Cap events for faster trials")
    #parser.add_argument("--num-workers", type=int, default=0)
    #parser.add_argument("--device", default=None, help="Device (default: auto)")
    #parser.add_argument("--angular-bias", action="store_true", help="Use AttentionBiasTransformer")
    #parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    #parser.add_argument("--study-name", default="particleman_hpo")
    #parser.add_argument("--storage", default="sqlite:///hpo_study.db", help="Optuna storage URL")
    #args = parser.parse_args()

    print(cfg)
    print(f"max_events: {cfg.data.max_events}")

    trials = cfg.get("trials", 20)
    study_name = cfg.get("study-name", "particleman_hpo")
    storage = cfg.get("storage", "sqlite:///hpo_study.db")

    # Setup distributed environment (returns rank=0, world_size=1 for single GPU)
    rank, world_size, device = setup_distributed(backend=cfg.distributed.backend)
    is_distributed = world_size > 1

    storage = optuna.storages.RDBStorage(
        url=storage,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )

    if rank == 0:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )
        #pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),

        n_existing = len(study.trials)
        if n_existing:
            logger.info(f"Resuming study '{study_name}' with {n_existing} existing trials")

        logger.info(
            f"Starting HPO: {trials} trials, {cfg.training.epochs} epochs each, "
            f"mode={cfg.training.mode}, data={cfg.data}"
        )
    dist.barrier()

    study = optuna.load_study(study_name=study_name, storage=storage)

    objective = build_objective(cfg)
    study.optimize(objective, 
                   n_trials=trials, 
                   show_progress_bar=True)

    if rank == 0:
        best = study.best_trial
        logger.info("=" * 60)
        logger.info(f"Best trial: #{best.number}")
        logger.info(f"  Val loss : {best.value:.4f}")
        logger.info("  Params   :")
        for k, v in best.params.items():
            logger.info(f"    {k}: {v}")
        logger.info("=" * 60)

    #storage.engine.dispose()
    cleanup_distributed()


if __name__ == "__main__":
    main()
