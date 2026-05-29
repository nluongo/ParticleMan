#!/usr/bin/env python3
"""
Evaluate a trained ParticleMan checkpoint and save masked-particle predictions.

Runs a single forward pass over a dataset split, collecting the predicted and true
values for every masked particle. Saves a compressed NumPy archive that can be
loaded later to produce histograms, filtered by particle_id, etc.

Usage:
    python scripts/eval_predictions.py \\
        ++checkpoint=outputs/ParticleMan/run/checkpoints/best_model.pt \\
        ++out_file=predictions.npz \\
        ++eval_split=val

    # Override data or batch size alongside eval params:
    python scripts/eval_predictions.py \\
        ++checkpoint=... \\
        data.max_events=5000 \\
        training.batch_size=64

Output .npz keys (one entry per masked particle):
    target_pt, pred_pt         — raw pT values
    target_eta, pred_eta       — raw eta values
    target_phi, pred_phi       — raw phi values (radians, wrapped to [-pi, pi])
    target_particle_id         — true integer class (0-14)
    pred_particle_id           — argmax of predicted logits
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from particleman.config import TrainConfig
from particleman.data import create_dataloaders
from particleman.models import (
    AttentionBiasTransformer,
    ParticleConfig,
    ParticleTransformer,
    PhiEncoding,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config_schema", node=TrainConfig)


def _denormalize(pred_pt, pred_eta, pred_phi, model_config: ParticleConfig):
    """Invert the normalization applied in ParticleTransformer._normalize_features."""
    pt_lo, pt_hi = model_config.pt_range
    eta_lo, eta_hi = model_config.eta_range
    phi_lo, phi_hi = model_config.phi_range

    raw_pt = pred_pt * (pt_hi - pt_lo) + pt_lo
    raw_eta = (pred_eta + 1) / 2 * (eta_hi - eta_lo) + eta_lo

    if model_config.phi_encoding == PhiEncoding.RAW:
        raw_phi = (pred_phi + 1) / 2 * (phi_hi - phi_lo) + phi_lo
    else:
        # SINCOS / NONE: predictions are already in wrapped-radian space
        raw_phi = pred_phi

    return raw_pt, raw_eta, raw_phi


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_path = OmegaConf.select(cfg, "checkpoint", default="")
    out_file = OmegaConf.select(cfg, "out_file", default="predictions.npz")
    split = OmegaConf.select(cfg, "eval_split", default="val")

    if not checkpoint_path:
        logger.error(
            "checkpoint must be specified, e.g. ++checkpoint=path/to/model.pt"
        )
        sys.exit(1)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Split: {split}")
    logger.info(f"Output: {out_file}")

    # Load checkpoint — model config is stored in the checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config: ParticleConfig = checkpoint["config"]

    logger.info(
        f"Model: d_model={model_config.d_model}, n_heads={model_config.n_heads}, "
        f"n_layers={model_config.n_layers}, phi_encoding={model_config.phi_encoding}"
    )

    if model_config.angular_attention_bias:
        model = AttentionBiasTransformer(model_config)
    else:
        model = ParticleTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create dataloaders (single-GPU only)
    max_events = cfg.data.get("max_events", None)
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers

    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.data,
        batch_size=batch_size,
        num_workers=num_workers,
        max_events=max_events,
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if split not in loaders:
        logger.error(f"Unknown split '{split}'. Choose from: train, val, test")
        sys.exit(1)
    dataloader = loaders[split]
    logger.info(f"Batches in '{split}' split: {len(dataloader)}")

    # Accumulate masked-particle predictions
    buckets: dict = {k: [] for k in [
        "target_pt", "pred_pt",
        "target_eta", "pred_eta",
        "target_phi", "pred_phi",
        "target_particle_id", "pred_particle_id",
    ]}

    max_mask_attempts = 5
    skipped = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                logger.info(f"  Batch {batch_idx}/{len(dataloader)}")

            pt = batch["pt"].to(device)
            eta = batch["eta"].to(device)
            phi = batch["phi"].to(device)
            particle_id = batch["particle_id"].to(device)
            padding_mask = ~batch["mask"].to(device)

            # Retry mask creation to avoid empty-mask batches
            mask_result = None
            for _ in range(max_mask_attempts):
                masked_inputs, mask_targets = model.create_masks(
                    pt, eta, phi, particle_id, padding_mask=padding_mask
                )
                if mask_targets["mask"].sum() > 0:
                    mask_result = (masked_inputs, mask_targets)
                    break

            if mask_result is None:
                skipped += 1
                continue

            masked_inputs, mask_targets = mask_result
            predictions = model(
                masked_inputs["pt"],
                masked_inputs["eta"],
                masked_inputs["phi"],
                masked_inputs["particle_id"],
                padding_mask=padding_mask,
            )

            mask = mask_targets["mask"]  # (B, S) bool

            # Targets are raw values
            buckets["target_pt"].append(mask_targets["pt"][mask].cpu())
            buckets["target_eta"].append(mask_targets["eta"][mask].cpu())
            buckets["target_phi"].append(mask_targets["phi"][mask].cpu())
            buckets["target_particle_id"].append(mask_targets["particle_id"][mask].cpu())

            # Denormalize continuous predictions back to physical units
            raw_pt, raw_eta, raw_phi = _denormalize(
                predictions["pt"][mask].cpu(),
                predictions["eta"][mask].cpu(),
                predictions["phi"][mask].cpu(),
                model_config,
            )
            buckets["pred_pt"].append(raw_pt)
            buckets["pred_eta"].append(raw_eta)
            buckets["pred_phi"].append(raw_phi)

            # Discrete prediction: argmax of logits
            buckets["pred_particle_id"].append(
                predictions["particle_id"][mask].argmax(dim=-1).cpu()
            )

    if skipped:
        logger.warning(
            f"Skipped {skipped} batches (no masked particles after {max_mask_attempts} attempts)"
        )

    if not any(buckets["target_pt"]):
        logger.error("No data collected — check that the dataloader returned batches")
        sys.exit(1)

    arrays = {k: torch.cat(v).numpy() for k, v in buckets.items()}
    total = len(arrays["target_pt"])
    logger.info(f"Total masked particles collected: {total:,}")

    out_path = Path(out_file)
    np.savez_compressed(out_path, **arrays)
    logger.info(f"Saved to {out_path}")

    logger.info("Summary:")
    logger.info(
        f"  target_pt : min={arrays['target_pt'].min():.2f}  "
        f"max={arrays['target_pt'].max():.2f}  mean={arrays['target_pt'].mean():.2f}"
    )
    logger.info(
        f"  pred_pt   : min={arrays['pred_pt'].min():.2f}  "
        f"max={arrays['pred_pt'].max():.2f}  mean={arrays['pred_pt'].mean():.2f}"
    )
    unique_ids, counts = np.unique(arrays["target_particle_id"], return_counts=True)
    logger.info(
        f"  particle_id counts: { {int(k): int(v) for k, v in zip(unique_ids, counts)} }"
    )


if __name__ == "__main__":
    main()
