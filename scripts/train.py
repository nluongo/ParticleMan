#!/usr/bin/env python3
"""
Training script for ParticleMan with Hydra configuration.

Supports both single-GPU and distributed multi-GPU/multi-node training.
Configuration is managed via YAML files and command-line overrides.

Usage:
    # Use default config (mc20_ttbar)
    python scripts/train_mc20.py
    
    # Override values via CLI (note: key=value syntax)
    python scripts/train_mc20.py training.epochs=20 model.d_model=256
    
    # Use different sub-configs
    python scripts/train_mc20.py data=physlite model=default
    
    # Override data settings
    python scripts/train_mc20.py data.max_events=5000 data.preprocessing.pt_cut=10.0
    
    # Print resolved config without running
    python scripts/train_mc20.py --cfg job
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 scripts/train_mc20.py training.batch_size=64
    
    # Multi-node with torchrun
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<master_ip> \\
        scripts/train_mc20.py
"""

from dataclasses import asdict
import logging
import sys
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Add src to path if needed
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import torch

from particleman.config import TrainConfig
from particleman.data import create_dataloaders, create_datasets
from particleman.models import ParticleTransformer, AttentionBiasTransformer, ParticleConfig, EmbeddingType, PhiEncoding
from particleman.training import (
    ParticleTrainer,
    OutputManager,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    create_distributed_dataloader,
)
from particleman.loggers import create_logger

# Setup basic console logging (file logging added later by OutputManager)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Register structured configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config_schema", node=TrainConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: TrainConfig) -> None:
    """Main training function."""
    # Setup distributed environment (returns rank=0, world_size=1 for single GPU)
    rank, world_size, device = setup_distributed(backend=cfg.distributed.backend)
    is_distributed = world_size > 1
    
    # Log resolved configuration (only on main process)
    if is_main_process(rank):
        logger.info(f"Resolved configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output manager only on main process
    output = None
    if is_main_process(rank):
        print("=" * 70)
        print("ParticleMan Training")
        print("=" * 70)
        
        output = OutputManager(
            experiment_name=cfg.output.experiment_name,
            run_name=cfg.output.run_name,
            base_dir=cfg.output.output_dir,
        )
        
        # Setup file logging
        output.setup_logging()
        
        # Save job info if running in PBS/SLURM
        output.save_job_info()
        
        logger.info(f"Experiment: {output.experiment_name}")
        logger.info(f"Run: {output.run_name}")
        logger.info(f"Output directory: {output.run_dir}")
        if is_distributed:
            logger.info(f"Distributed training: world_size={world_size}")
        logger.info(f"Device: {device}")

    # Log data info
    if is_main_process(rank):
        source = cfg.data.get("source", {})
        preproc = cfg.data.get("preprocessing", {})
        collections = cfg.data.get("collections", {})
        enabled_collections = [name for name, c in collections.items() if c.get("enabled", True)]
        
        logger.info(f"Source type: {source.get('type', 'root')}")
        logger.info(f"Files: {source.get('files', [])}")
        logger.info(f"Enabled collections: {enabled_collections}")
        logger.info(f"Max particles per event: {preproc.get('max_particles', 200)}")
    
    # Create dataloaders
    if is_main_process(rank):
        logger.info("Creating dataloaders...")
    
    # Get max_events from config (can be None for all events)
    max_events = cfg.data.get("max_events", None)
    
    try:
        if is_distributed:
            # For distributed training, create datasets then wrap with distributed sampler
            train_dataset, val_dataset, test_dataset = create_datasets(
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
            
            test_loader = create_distributed_dataloader(
                test_dataset,
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
            train_loader, val_loader, test_loader = create_dataloaders(
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
        logger.info(f"Test batches: {len(test_loader)}")
        if is_distributed:
            logger.info(f"Effective batch size: {cfg.training.batch_size * world_size}")
    
    # Get a sample batch to verify data (only on main process)
    if is_main_process(rank):
        logger.info("Verifying data loading...")
        sample_batch = next(iter(train_loader))
        logger.info(f"Batch shapes:")
        logger.info(f"  pt: {sample_batch['pt'].shape}")
        logger.info(f"  eta: {sample_batch['eta'].shape}")
        logger.info(f"  phi: {sample_batch['phi'].shape}")
        logger.info(f"  particle_id: {sample_batch['particle_id'].shape}")
        logger.info(f"  mask: {sample_batch['mask'].shape}")
        
        # Check particle statistics
        n_particles = sample_batch['mask'].sum(dim=1).float()
        logger.info(f"Particles per event: mean={n_particles.mean():.1f}, min={n_particles.min():.0f}, max={n_particles.max():.0f}")
    
    # Create model configuration
    embedding_type = EmbeddingType.JOINT if cfg.model.embedding == "joint" else EmbeddingType.CONCAT
    if cfg.model.phi_encoding == "sincos":
        phi_encoding = PhiEncoding.SINCOS
    elif cfg.model.phi_encoding == "none":
        phi_encoding = PhiEncoding.NONE
    else:
        phi_encoding = PhiEncoding.RAW
    
    # Get preprocessing values from data config
    preproc = cfg.data.get("preprocessing", {})
    max_particles = preproc.get("max_particles", 200)
    eta_cut = preproc.get("eta_cut", 5.0)
    
    model_config = ParticleConfig(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_model * 4,
        dropout=cfg.model.dropout,
        max_particles=max_particles,
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
    )
    model_config_dict = asdict(model_config)
    
    if is_main_process(rank):
        logger.info(f"Model configuration:")
        logger.info(f"  Embedding type: {model_config.embedding_type.value}")
        logger.info(f"  Phi encoding: {model_config.phi_encoding.value}")
        logger.info(f"  d_model: {model_config.d_model}")
        logger.info(f"  n_heads: {model_config.n_heads}")
        logger.info(f"  n_layers: {model_config.n_layers}")
        logger.info(f"  mask_prob: {model_config.mask_prob}")
    
    # Create model
    if is_main_process(rank):
        logger.info("Creating model...")
    if cfg.model.angular_attention_bias:
        model = AttentionBiasTransformer(model_config)
    else:
        model = ParticleTransformer(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main_process(rank):
        logger.info(f"Model parameters: {n_params:,}")
    
    # Build hyperparameters dict for logging
    training_config_dict = {
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
        "train_dataset_size": len(train_loader.dataset),
        "val_dataset_size": len(val_loader.dataset),
        "test_dataset_size": len(test_loader.dataset),
    }
    
    # Save config to output directory (only on main process)
    if is_main_process(rank):
        # Save the full Hydra config as well
        output.save_config({
            "model": model_config_dict,
            "training": training_config_dict,
            "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        })
    
    # Create experiment logger with same run_name (only on main process)
    train_experiment_logger = None
    if is_main_process(rank):
        train_experiment_logger = create_logger(
            cfg.output.experiment_logger,
            experiment_name=cfg.output.experiment_name,
            run_name=output.run_name,
        )
        # Log hyperparameters to experiment logger
        train_experiment_logger.log_params(model_config_dict)
        train_experiment_logger.log_params(training_config_dict)

    # Create trainer
    if is_main_process(rank):
        logger.info("Creating trainer...")
    
    save_dir = output.checkpoint_dir if output else None
    
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
    )
    
    if is_main_process(rank):
        logger.info(f"Training device: {trainer.device}")
        logger.info(f"Checkpoint directory: {save_dir}")
    
    # Train
    if is_main_process(rank):
        logger.info(f"\nStarting training for {cfg.training.epochs} epochs...")
        logger.info("=" * 70)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_main_process(rank):
            logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
    except Exception as e:
        if is_main_process(rank):
            logger.error(f"Training failed: {e}")
        raise
    finally:
        if train_experiment_logger:
            train_experiment_logger.close()
    
    # Final evaluation on test set (all processes participate to avoid timeout)
    if is_main_process(rank):
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating on test set...")
    
    # All processes run evaluation (trainer.evaluate handles reduction across processes)
    test_losses = trainer.evaluate(test_loader)
    
    # Only main process logs results and saves
    if is_main_process(rank):
        test_experiment_logger = create_logger(
            cfg.output.experiment_logger,
            experiment_name=f"{cfg.output.experiment_name}_test",
            run_name=output.run_name,
        )
        test_experiment_logger.log_params(model_config_dict)
        test_experiment_logger.log_params(training_config_dict)
        
        logger.info(f"Test Results:")
        logger.info(f"  Total Loss: {test_losses['total_loss']:.4f}")
        logger.info(f"  pT Loss: {test_losses['pt_loss']:.4f}")
        logger.info(f"  eta Loss: {test_losses['eta_loss']:.4f}")
        logger.info(f"  phi Loss: {test_losses['phi_loss']:.4f}")
        logger.info(f"  Particle ID Loss: {test_losses['particle_id_loss']:.4f}")
        
        test_experiment_logger.log_metrics(test_losses, step=0)
        test_experiment_logger.close()
        
        # Save summary
        output.save_summary(test_losses)
        
        logger.info("\n" + "=" * 70)
        logger.info("Training complete!")
        logger.info(f"All outputs saved to: {output.run_dir}")
        logger.info("=" * 70)
    
    # Clean up distributed environment
    cleanup_distributed()


if __name__ == "__main__":
    main()

