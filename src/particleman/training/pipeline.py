"""End-to-end training run: data, model, trainer, test evaluation.

``run_training`` is what ``scripts/train.py`` (and the ``train`` mode of
``scripts/run.py``) calls with the composed Hydra config. It works for a
single process and for every rank of a distributed launch.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from particleman.data import create_dataloaders, create_datasets
from particleman.loggers import create_logger
from particleman.models import (
    AttentionBiasTransformer,
    EmbeddingType,
    ParticleConfig,
    ParticleTransformer,
    PhiEncoding,
)
from particleman.training.distributed import (
    cleanup_distributed,
    create_distributed_dataloader,
    is_main_process,
    setup_distributed,
)
from particleman.training.output_manager import OutputManager
from particleman.training.trainer import ParticleTrainer

logger = logging.getLogger(__name__)


def build_model_config(cfg: DictConfig) -> ParticleConfig:
    """Build the ``ParticleConfig`` from the ``model``/``training``/``data`` configs."""
    embedding_type = (
        EmbeddingType.JOINT if cfg.model.embedding == "joint" else EmbeddingType.CONCAT
    )
    if cfg.model.phi_encoding == "sincos":
        phi_encoding = PhiEncoding.SINCOS
    elif cfg.model.phi_encoding == "none":
        phi_encoding = PhiEncoding.NONE
    else:
        phi_encoding = PhiEncoding.RAW

    preproc = cfg.data.get("preprocessing", {})
    eta_cut = preproc.get("eta_cut", 5.0)

    return ParticleConfig(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_particles=preproc.get("max_particles", 200),
        n_particle_types=16,  # 0-15 particle categories
        mask_prob=cfg.training.mask_prob,
        embedding_type=embedding_type,
        phi_encoding=phi_encoding,
        # Feature ranges based on preprocessing
        pt_range=(0.0, 500.0),  # pT in GeV after scaling
        eta_range=(-eta_cut, eta_cut),
        phi_range=(-3.14159, 3.14159),
        angular_attention_bias=cfg.model.angular_attention_bias,
        bias_hidden_dim=cfg.model.bias_hidden_dim,
        n_event_types=cfg.data.get("n_event_types", 0),
        event_label_mask_prob=cfg.training.event_label_mask_prob,
    )


def run_training(cfg: DictConfig) -> None:
    """
    Train a model as described by the composed config, then evaluate it on
    the test split.

    Args:
        cfg: Composed root config (``configs/config.yaml`` or another root)
    """
    # Setup distributed environment (returns rank=0, world_size=1 for single GPU)
    rank, world_size, device = setup_distributed(
        backend=cfg.distributed.backend,
        timeout_s=cfg.distributed.get("timeout_s", 60),
    )
    is_distributed = world_size > 1
    main_process = is_main_process(rank)

    # Create output manager only on main process
    output: Optional[OutputManager] = None
    if main_process:
        logger.info(f"Resolved configuration:\n{OmegaConf.to_yaml(cfg)}")
        logger.info("=" * 70)
        logger.info("ParticleMan Training")
        logger.info("=" * 70)

        output = OutputManager(
            experiment_name=cfg.output.experiment_name,
            timestamp=cfg.output.timestamp,
            run_name=cfg.output.run_name,
            base_dir=cfg.output.output_dir,
        )
        output.setup_logging()
        # Save job info if running in PBS/SLURM
        output.save_job_info()

        logger.info(f"Experiment: {output.experiment_name}")
        logger.info(f"Run: {output.run_name}")
        logger.info(f"Output directory: {output.run_dir}")
        if is_distributed:
            logger.info(f"Distributed training: world_size={world_size}")
        logger.info(f"Device: {device}")

        source = cfg.data.get("source", {})
        preproc = cfg.data.get("preprocessing", {})
        collections = cfg.data.get("collections", {})
        enabled_collections = [
            name for name, c in collections.items() if c.get("enabled", True)
        ]
        logger.info(f"Source type: {source.get('type', 'root')}")
        logger.info(f"Files: {source.get('files', [])}")
        logger.info(f"Enabled collections: {enabled_collections}")
        logger.info(f"Max particles per event: {preproc.get('max_particles', 200)}")
        logger.info("Creating dataloaders...")

    # Get max_events from config (can be None for all events)
    max_events = cfg.data.get("max_events", None)

    try:
        if is_distributed:
            # Create datasets, then wrap them with a distributed sampler
            train_dataset, val_dataset, test_dataset = create_datasets(
                cfg.data,
                max_events=max_events,
            )
            seed = cfg.data.get("split", {}).get("seed", 42)
            loaders = [
                create_distributed_dataloader(
                    dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=shuffle,
                    num_workers=cfg.training.num_workers,
                    rank=rank,
                    world_size=world_size,
                    drop_last=shuffle,
                    seed=seed,
                )
                for dataset, shuffle in (
                    (train_dataset, True),
                    (val_dataset, False),
                    (test_dataset, False),
                )
            ]
            train_loader, val_loader, test_loader = loaders
        else:
            train_loader, val_loader, test_loader = create_dataloaders(
                cfg.data,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.training.num_workers,
                max_events=max_events,
            )
    except Exception as e:
        if main_process:
            logger.error(f"Failed to create dataloaders: {e}")
            logger.error("Make sure the data files exist and the config is correct.")
        cleanup_distributed()
        raise

    if main_process:
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        if is_distributed:
            logger.info(f"Effective batch size: {cfg.training.batch_size * world_size}")

        # Get a sample batch to verify data
        logger.info("Verifying data loading...")
        sample_batch = next(iter(train_loader))
        logger.info("Batch shapes:")
        for key in ("pt", "eta", "phi", "particle_id", "mask"):
            logger.info(f"  {key}: {sample_batch[key].shape}")
        n_particles = sample_batch["mask"].sum(dim=1).float()
        logger.info(
            f"Particles per event: mean={n_particles.mean():.1f}, "
            f"min={n_particles.min():.0f}, max={n_particles.max():.0f}"
        )

    model_config = build_model_config(cfg)
    model_config_dict = asdict(model_config)

    if main_process:
        logger.info("Model configuration:")
        logger.info(f"  Embedding type: {model_config.embedding_type.value}")
        logger.info(f"  Phi encoding: {model_config.phi_encoding.value}")
        logger.info(f"  d_model: {model_config.d_model}")
        logger.info(f"  d_ff: {model_config.d_ff}")
        logger.info(f"  n_heads: {model_config.n_heads}")
        logger.info(f"  n_layers: {model_config.n_layers}")
        logger.info(f"  mask_prob: {model_config.mask_prob}")
        logger.info("Creating model...")

    model: ParticleTransformer
    if cfg.model.angular_attention_bias:
        model = AttentionBiasTransformer(model_config)
    else:
        model = ParticleTransformer(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    if main_process:
        logger.info(f"Model parameters: {n_params:,}")

    # Hyperparameters for logging
    training_config_dict: Dict[str, Any] = {
        "trainer_class": "ParticleTrainer",
        "max_events": max_events,
        "batch_size": cfg.training.batch_size,
        "effective_batch_size": cfg.training.batch_size * world_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "epochs": cfg.training.epochs,
        "num_params": n_params,
        "world_size": world_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "train_dataset_size": len(train_loader.dataset),  # type: ignore[arg-type]
        "val_dataset_size": len(val_loader.dataset),  # type: ignore[arg-type]
        "test_dataset_size": len(test_loader.dataset),  # type: ignore[arg-type]
    }

    # Save config and create the experiment logger (main process only)
    train_experiment_logger = None
    if output is not None:
        output.save_config(
            {
                "model": model_config_dict,
                "training": training_config_dict,
                "hydra_config": OmegaConf.to_container(
                    cfg, resolve=True, enum_to_str=True
                ),
            }
        )
        train_experiment_logger = create_logger(
            cfg.output.experiment_logger,
            experiment_name=cfg.output.experiment_name,
            run_name=output.run_name,
        )
        train_experiment_logger.log_params(model_config_dict)
        train_experiment_logger.log_params(training_config_dict)
        logger.info("Creating trainer...")

    save_dir = output.checkpoint_dir if output else None
    profile_dir = output.run_dir / "profiler" if output else None
    pretrained = cfg.training.pretrained_checkpoint

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
        log_interval=cfg.output.log_interval,
        save_dir=save_dir,
        experiment_logger=train_experiment_logger,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        profile=cfg.training.profile,
        profile_dir=profile_dir,
        pretrained_checkpoint=Path(pretrained) if pretrained else None,
        mode=cfg.training.mode,
        timing_path=output.run_dir / "timing.json" if output else None,
    )

    if main_process:
        logger.info(f"Training device: {trainer.device}")
        logger.info(f"Checkpoint directory: {save_dir}")
        logger.info(f"\nStarting training for {cfg.training.epochs} epochs...")
        logger.info("=" * 70)

    try:
        trainer.train()
    except KeyboardInterrupt:
        if main_process:
            logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
    except Exception as e:
        if main_process:
            logger.error(f"Training failed: {e}")
        raise
    finally:
        if train_experiment_logger:
            train_experiment_logger.close()

    # Final evaluation on the test set (all processes participate to avoid timeout)
    if main_process:
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating on test set...")
    test_losses = trainer.evaluate(test_loader)

    if output is not None:
        test_experiment_logger = create_logger(
            cfg.output.experiment_logger,
            experiment_name=f"{cfg.output.experiment_name}_test",
            run_name=output.run_name,
        )
        test_experiment_logger.log_params(model_config_dict)
        test_experiment_logger.log_params(training_config_dict)

        logger.info("Test Results:")
        logger.info(f"  Total Loss: {test_losses['total_loss']:.4f}")
        logger.info(f"  pT Loss: {test_losses['pt_loss']:.4f}")
        logger.info(f"  eta Loss: {test_losses['eta_loss']:.4f}")
        logger.info(f"  phi Loss: {test_losses['phi_loss']:.4f}")
        logger.info(f"  Particle ID Loss: {test_losses['particle_id_loss']:.4f}")

        test_experiment_logger.log_metrics(test_losses, step=0)
        test_experiment_logger.close()

        output.save_summary(test_losses, timing=trainer.timing)

        logger.info("\n" + "=" * 70)
        logger.info("Training complete!")
        logger.info(f"All outputs saved to: {output.run_dir}")
        logger.info("=" * 70)

    cleanup_distributed()
