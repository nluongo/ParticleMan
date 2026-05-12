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

import argparse
import logging
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import optuna
from omegaconf import OmegaConf

from particleman.data import create_dataloaders
from particleman.models import (
    AttentionBiasTransformer,
    EmbeddingType,
    ParticleConfig,
    ParticleTransformer,
    PhiEncoding,
)
from particleman.training import ParticleTrainer

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


def build_objective(data_cfg: dict, args: argparse.Namespace):
    """Return an Optuna objective function closed over data and CLI args."""

    # Load dataloaders once — shared across all trials to avoid repeated I/O.
    logger.info("Loading data (shared across trials)...")
    max_events = getattr(args, "max_events", None)
    train_loader, val_loader, _ = create_dataloaders(
        data_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_events=max_events,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    preproc = data_cfg.get("preprocessing", {})
    max_particles = preproc.get("max_particles", 200)
    eta_cut = preproc.get("eta_cut", 5.0)
    n_event_types = data_cfg.get("n_event_types", 0)

    def objective(trial: optuna.Trial) -> float:
        # --- Suggest model-size hyperparameters ---
        d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
        # n_heads must divide d_model evenly
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        n_heads = trial.suggest_categorical("n_heads", valid_heads)
        n_layers = trial.suggest_int("n_layers", 2, 8)
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        d_model_ff_mult = trial.suggest_categorical("d_model_ff_mult", [1, 2, 4, 8])

        model_config = ParticleConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * d_model_ff_mult,
            dropout=dropout,
            max_particles=max_particles,
            n_particle_types=16,
            mask_prob=args.mask_prob,
            embedding_type=EmbeddingType.JOINT,
            phi_encoding=PhiEncoding.RAW,
            pt_range=(0.0, 500.0),
            eta_range=(-eta_cut, eta_cut),
            phi_range=(-3.14159, 3.14159),
            n_event_types=n_event_types,
            event_label_mask_prob=args.mask_prob,
        )

        n_params = sum(
            p.numel()
            for p in (
                ParticleTransformer(model_config) if not args.angular_bias
                else AttentionBiasTransformer(model_config)
            ).parameters()
        )
        logger.info(
            f"Trial {trial.number}: d_model={d_model}, n_heads={n_heads}, "
            f"n_layers={n_layers}, dropout={dropout:.2f} — {n_params:,} params"
        )

        model = (
            AttentionBiasTransformer(model_config)
            if args.angular_bias
            else ParticleTransformer(model_config)
        )

        trainer = ParticleTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            log_interval=999_999,  # suppress per-step logging
            save_dir=None,
            mode=args.mode,
        )

        best_val_loss = float("inf")
        for epoch in range(args.epochs):
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
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                raise optuna.TrialPruned()

        logger.info(f"Trial {trial.number} finished — best val loss: {best_val_loss:.4f}")
        return best_val_loss

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="ParticleMan HPO with Optuna")
    parser.add_argument("--data-config", default="default", help="Data config name (without .yaml)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mode", default="pretrain", choices=["pretrain", "classify"])
    parser.add_argument("--max-events", type=int, default=None, help="Cap events for faster trials")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--angular-bias", action="store_true", help="Use AttentionBiasTransformer")
    parser.add_argument("--study-name", default="particleman_hpo")
    parser.add_argument("--storage", default="sqlite:///hpo_study.db", help="Optuna storage URL")
    args = parser.parse_args()

    data_cfg = load_data_config(args.data_config)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing:
        logger.info(f"Resuming study '{args.study_name}' with {n_existing} existing trials")

    logger.info(
        f"Starting HPO: {args.trials} trials, {args.epochs} epochs each, "
        f"mode={args.mode}, data={args.data_config}"
    )

    objective = build_objective(data_cfg, args)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    logger.info("=" * 60)
    logger.info(f"Best trial: #{best.number}")
    logger.info(f"  Val loss : {best.value:.4f}")
    logger.info("  Params   :")
    for k, v in best.params.items():
        logger.info(f"    {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
