#!/usr/bin/env python
"""
Trace Training Pipeline

This script provides complete visibility into every step of the training process.
It loads a single sample, processes it through each stage, and prints detailed
information about what happens at each step.

Usage:
    python scripts/trace_training_pipeline.py --config configs/data/preprocessed_h5.yaml
    python scripts/trace_training_pipeline.py --config configs/data/mc20_ttbar.yaml --event-idx 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from particleman.data.config import DataConfig
from particleman.data.dataset import ParticleDataset
from particleman.data.hdf5_loader import HDF5ParticleLoader
from particleman.data.root_loader import ROOTParticleLoader
from particleman.models.particle_transformer import (
    ParticleTransformer,
    ParticleConfig,
    EmbeddingType,
    PhiEncoding,
)


def print_header(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def print_array_stats(name: str, arr) -> None:
    """Print statistics about an array/tensor."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    
    if arr.size == 0:
        print(f"  {name}: empty array")
        return
    
    if arr.dtype == bool:
        n_true = arr.sum()
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, True={n_true}, False={arr.size - n_true}")
    elif np.issubdtype(arr.dtype, np.integer):
        unique = np.unique(arr)
        if len(unique) <= 10:
            print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, unique={unique.tolist()}")
        else:
            print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, n_unique={len(unique)}")
    else:
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}")


def trace_config_loading(config_path: str) -> DataConfig:
    """Trace the configuration loading step."""
    print_header("STEP 1: Configuration Loading")
    
    print(f"Loading config from: {config_path}")
    config = DataConfig.from_yaml(config_path)
    
    print_subheader("Source Configuration")
    print(f"  Source type: {config.source.type}")
    print(f"  Files: {config.source.files}")
    print(f"  Tree name: {config.source.tree_name}")
    
    print_subheader("Enabled Collections")
    for name, coll in config.get_enabled_collections().items():
        print(f"  {name}:")
        print(f"    - pt column: {coll.columns.pt}")
        print(f"    - eta column: {coll.columns.eta}")
        print(f"    - phi column: {coll.columns.phi}")
        print(f"    - particle_id column: {coll.columns.particle_id}")
        print(f"    - fixed_particle_id: {coll.fixed_particle_id}")
        print(f"    - is_vector: {coll.is_vector}")
    
    print_subheader("Preprocessing Configuration")
    print(f"  pt_cut: {config.preprocessing.pt_cut} GeV")
    print(f"  eta_cut: {config.preprocessing.eta_cut}")
    print(f"  max_particles: {config.preprocessing.max_particles}")
    print(f"  pt_scale: {config.preprocessing.pt_scale}")
    print(f"  shuffle_particles: {config.preprocessing.shuffle_particles}")
    
    print_subheader("Particle ID Map")
    for raw_id, mapped_id in config.particle_id_map.items():
        print(f"  {raw_id} -> {mapped_id}")
    print(f"  default_particle_id: {config.default_particle_id}")
    
    print_subheader("Split Configuration")
    print(f"  train: {config.split.train}")
    print(f"  val: {config.split.val}")
    print(f"  test: {config.split.test}")
    print(f"  seed: {config.split.seed}")
    
    return config


def trace_data_loading(config: DataConfig, event_idx: int) -> dict:
    """Trace the raw data loading step."""
    print_header("STEP 2: Raw Data Loading")
    
    # Create the appropriate loader
    if config.source.type == "hdf5":
        print("Creating HDF5ParticleLoader...")
        loader = HDF5ParticleLoader(config)
        print(f"  Available datasets: {loader.get_dataset_names()[:10]}...")  # First 10
    else:
        print("Creating ROOTParticleLoader...")
        loader = ROOTParticleLoader(config)
    
    print(f"\nTotal events in loader: {len(loader)}")
    print(f"Requesting event index: {event_idx}")
    
    # Get the raw event
    event = loader.get_event(event_idx)
    
    print_subheader("Loaded Event Data")
    for key, value in event.items():
        if isinstance(value, np.ndarray):
            print_array_stats(key, value)
        else:
            print(f"  {key}: {value}")
    
    return event


def trace_preprocessing(config: DataConfig, event_idx: int) -> tuple:
    """Trace preprocessing in detail by stepping through base_loader logic."""
    print_header("STEP 3: Preprocessing Pipeline (Detailed)")
    
    # Create loader
    if config.source.type == "hdf5":
        loader = HDF5ParticleLoader(config)
    else:
        loader = ROOTParticleLoader(config)
    
    print_subheader("Step 3a: Load Each Collection")
    
    all_pt = []
    all_eta = []
    all_phi = []
    all_particle_id = []
    
    for name, coll_cfg in loader.enabled_collections.items():
        print(f"\n  Loading collection: {name}")
        pt, eta, phi, pid = loader._load_collection(event_idx, name, coll_cfg)
        
        print(f"    Particles loaded: {len(pt)}")
        if len(pt) > 0:
            print(f"    pt range: [{pt.min():.2f}, {pt.max():.2f}]")
            print(f"    eta range: [{eta.min():.2f}, {eta.max():.2f}]")
            print(f"    phi range: [{phi.min():.2f}, {phi.max():.2f}]")
            print(f"    particle_ids: {np.unique(pid).tolist()}")
            
            all_pt.append(pt)
            all_eta.append(eta)
            all_phi.append(phi)
            all_particle_id.append(pid)
    
    # Combine
    if all_pt:
        pt = np.concatenate(all_pt)
        eta = np.concatenate(all_eta)
        phi = np.concatenate(all_phi)
        particle_id = np.concatenate(all_particle_id)
    else:
        pt = np.array([], dtype=np.float32)
        eta = np.array([], dtype=np.float32)
        phi = np.array([], dtype=np.float32)
        particle_id = np.array([], dtype=np.int32)
    
    print_subheader("Step 3b: Combined Raw Data (Before Preprocessing)")
    print(f"  Total particles: {len(pt)}")
    if len(pt) > 0:
        print_array_stats("pt (raw)", pt)
        print_array_stats("eta", eta)
        print_array_stats("phi", phi)
        print_array_stats("particle_id", particle_id)
    
    print_subheader("Step 3c: Apply Preprocessing")
    cfg = config.preprocessing
    
    # Scale pT
    print(f"\n  Scaling pT by 1/{cfg.pt_scale}...")
    pt_scaled = pt / cfg.pt_scale
    if len(pt_scaled) > 0:
        print(f"    Before: pt range [{pt.min():.2f}, {pt.max():.2f}]")
        print(f"    After:  pt range [{pt_scaled.min():.4f}, {pt_scaled.max():.4f}]")
    
    # Apply cuts
    print(f"\n  Applying cuts: pt >= {cfg.pt_cut} GeV, |eta| <= {cfg.eta_cut}...")
    n_before = len(pt_scaled)
    mask = (pt_scaled >= cfg.pt_cut) & (np.abs(eta) <= cfg.eta_cut)
    pt_cut = pt_scaled[mask]
    eta_cut = eta[mask]
    phi_cut = phi[mask]
    pid_cut = particle_id[mask]
    n_after = len(pt_cut)
    print(f"    Particles before cuts: {n_before}")
    print(f"    Particles after cuts: {n_after}")
    print(f"    Particles removed: {n_before - n_after}")
    
    # Shuffle
    if cfg.shuffle_particles:
        print(f"\n  Shuffling particles (seed={config.split.seed})...")
        print(f"    Note: Order will be randomized")
    
    print_subheader("Step 3d: Pad/Truncate to Fixed Length")
    max_p = cfg.max_particles
    n_particles = len(pt_cut)
    print(f"  max_particles: {max_p}")
    print(f"  actual particles: {n_particles}")
    
    if n_particles > max_p:
        print(f"  Action: TRUNCATE (removing {n_particles - max_p} particles)")
    elif n_particles < max_p:
        print(f"  Action: PAD (adding {max_p - n_particles} padding tokens)")
    else:
        print(f"  Action: NONE (exact match)")
    
    # Get final processed event
    final_event = loader.get_event(event_idx)
    
    print_subheader("Step 3e: Final Preprocessed Event")
    for key, value in final_event.items():
        if isinstance(value, np.ndarray):
            print_array_stats(key, value)
        else:
            print(f"  {key}: {value}")
    
    # Show first few particles
    n_show = min(5, final_event["n_particles"])
    print(f"\n  First {n_show} real particles:")
    for i in range(n_show):
        print(f"    [{i}] pt={final_event['pt'][i]:.4f}, eta={final_event['eta'][i]:.4f}, "
              f"phi={final_event['phi'][i]:.4f}, id={final_event['particle_id'][i]}")
    
    return final_event, loader


def trace_dataset_conversion(config: DataConfig, event_idx: int) -> dict:
    """Trace PyTorch dataset conversion."""
    print_header("STEP 4: PyTorch Dataset Conversion")
    
    print("Creating ParticleDataset...")
    dataset = ParticleDataset(config)
    
    print(f"  Dataset length: {len(dataset)}")
    print(f"  max_particles: {dataset.max_particles}")
    
    print_subheader("Getting Sample from Dataset")
    print(f"  Requesting index: {event_idx}")
    
    sample = dataset[event_idx]
    
    print_subheader("Tensor Conversion Results")
    for key, tensor in sample.items():
        print(f"  {key}:")
        print(f"    dtype: {tensor.dtype}")
        print(f"    shape: {tensor.shape}")
        if tensor.numel() <= 10:
            print(f"    values: {tensor.tolist()}")
        else:
            print(f"    first 5: {tensor[:5].tolist()}")
            if tensor.dtype == torch.bool:
                print(f"    True count: {tensor.sum().item()}")
            elif tensor.dtype in [torch.float32, torch.float64]:
                print(f"    min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
    
    return sample


def trace_batching(sample: dict) -> dict:
    """Trace batching (simulated with batch size 1)."""
    print_header("STEP 5: Batching")
    
    print("Simulating batch creation (batch_size=1)...")
    
    batch = {}
    for key, tensor in sample.items():
        if tensor.dim() == 0:
            batch[key] = tensor.unsqueeze(0)
        else:
            batch[key] = tensor.unsqueeze(0)
    
    print_subheader("Batched Tensors")
    for key, tensor in batch.items():
        print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    return batch


def trace_masking(model: ParticleTransformer, batch: dict, device: torch.device, max_attempts: int = 10) -> tuple:
    """Trace the masking process in detail."""
    print_header("STEP 6: Masking for Pre-training")
    
    # Move to device
    pt = batch['pt'].to(device)
    eta = batch['eta'].to(device)
    phi = batch['phi'].to(device)
    particle_id = batch['particle_id'].to(device)
    real_particle_mask = batch['mask'].to(device)
    padding_mask = ~real_particle_mask
    
    print_subheader("Input to Masking")
    print(f"  Batch shape: {pt.shape}")
    print(f"  Real particles (mask=True): {real_particle_mask.sum().item()}")
    print(f"  Padding positions: {padding_mask.sum().item()}")
    print(f"  Mask probability: {model.config.mask_prob}")
    print(f"  Mask noise std: {model.config.mask_continuous_std}")
    
    # Create masks with retry logic if empty
    print_subheader("Creating Masks")
    n_real = real_particle_mask.sum().item()
    
    for attempt in range(max_attempts):
        torch.manual_seed(42 + attempt)  # Different seed each attempt
        masked_inputs, mask_targets = model.create_masks(
            pt, eta, phi, particle_id,
            padding_mask=padding_mask
        )
        
        prediction_mask = mask_targets['mask']
        n_masked = prediction_mask.sum().item()
        
        if n_masked > 0:
            if attempt > 0:
                print(f"  Successfully created mask on attempt {attempt + 1}")
            break
        print(f"  Attempt {attempt + 1}: No particles masked, resampling...")
    
    if n_masked == 0:
        raise RuntimeError(
            f"Failed to create non-empty mask after {max_attempts} attempts. "
            f"This event may have too few real particles ({n_real})."
        )
    
    print(f"  Particles selected for masking: {n_masked}")
    print(f"  Masking fraction: {n_masked / max(n_real, 1):.2%}")
    
    print_subheader("Masked Positions")
    masked_indices = prediction_mask[0].nonzero(as_tuple=True)[0].tolist()
    print(f"  Masked indices: {masked_indices[:20]}{'...' if len(masked_indices) > 20 else ''}")
    
    print_subheader("Original vs Masked Values (first 3 masked positions)")
    for i, idx in enumerate(masked_indices[:3]):
        print(f"\n  Position {idx}:")
        print(f"    pt:  original={pt[0, idx].item():.4f} -> masked={masked_inputs['pt'][0, idx].item():.4f}")
        print(f"    eta: original={eta[0, idx].item():.4f} -> masked={masked_inputs['eta'][0, idx].item():.4f}")
        print(f"    phi: original={phi[0, idx].item():.4f} -> masked={masked_inputs['phi'][0, idx].item():.4f}")
        print(f"    id:  original={particle_id[0, idx].item()} -> masked={masked_inputs['particle_id'][0, idx].item()} (mask token)")
    
    print_subheader("Mask Targets Summary")
    for key, tensor in mask_targets.items():
        if key != 'mask':
            print_array_stats(key, tensor.cpu())
    
    return masked_inputs, mask_targets, padding_mask


def trace_normalization(model: ParticleTransformer, masked_inputs: dict) -> tuple:
    """Trace feature normalization."""
    print_header("STEP 7: Feature Normalization")
    
    pt = masked_inputs['pt']
    eta = masked_inputs['eta']
    phi = masked_inputs['phi']
    
    config = model.config
    print_subheader("Normalization Ranges from Config")
    print(f"  pt_range: {config.pt_range} -> normalized to [0, 1]")
    print(f"  eta_range: {config.eta_range} -> normalized to [-1, 1]")
    print(f"  phi_range: {config.phi_range}")
    print(f"  phi_encoding: {config.phi_encoding}")
    
    print_subheader("Before Normalization")
    print_array_stats("pt", pt.cpu())
    print_array_stats("eta", eta.cpu())
    print_array_stats("phi", phi.cpu())
    
    # Perform normalization
    pt_norm, eta_norm, phi_processed = model._normalize_features(pt, eta, phi)
    
    print_subheader("After Normalization")
    print_array_stats("pt_norm", pt_norm.cpu())
    print_array_stats("eta_norm", eta_norm.cpu())
    
    if config.phi_encoding == PhiEncoding.SINCOS:
        print(f"  phi_processed: raw phi (sin/cos computed in embedding)")
        print_array_stats("phi_raw", phi_processed.cpu())
    else:
        print_array_stats("phi_norm", phi_processed.cpu())
    
    return pt_norm, eta_norm, phi_processed


def trace_embedding(model: ParticleTransformer, pt_norm, eta_norm, phi_processed, particle_id) -> torch.Tensor:
    """Trace the embedding layer."""
    print_header("STEP 8: Particle Embedding")
    
    config = model.config
    print_subheader("Embedding Configuration")
    print(f"  embedding_type: {config.embedding_type}")
    print(f"  phi_encoding: {config.phi_encoding}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_particle_types: {config.n_particle_types}")
    
    if config.embedding_type == EmbeddingType.JOINT:
        print(f"  id_embed_dim: {config.id_embed_dim}")
        print(f"  embed_hidden_dim: {config.embed_hidden_dim}")
    
    print_subheader("Embedding Layer Architecture")
    print(model.embedding)
    
    print_subheader("Input to Embedding")
    print(f"  pt_norm shape: {pt_norm.shape}")
    print(f"  eta_norm shape: {eta_norm.shape}")
    print(f"  phi_processed shape: {phi_processed.shape}")
    print(f"  particle_id shape: {particle_id.shape}")
    
    # Compute embeddings
    embeddings = model.embedding(pt_norm, eta_norm, phi_processed, particle_id)
    
    print_subheader("Embedding Output")
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  embeddings dtype: {embeddings.dtype}")
    print_array_stats("embeddings", embeddings.cpu())
    
    # Show embedding for first particle
    print(f"\n  First particle embedding (first 10 dims):")
    print(f"    {embeddings[0, 0, :10].tolist()}")
    
    return embeddings


def trace_transformer(model: ParticleTransformer, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Trace the transformer encoder."""
    print_header("STEP 9: Transformer Encoder")
    
    config = model.config
    print_subheader("Transformer Configuration")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  dropout: {config.dropout}")
    
    print_subheader("Transformer Architecture")
    print(model.transformer)
    
    print_subheader("Input to Transformer")
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  padding_mask shape: {padding_mask.shape}")
    print(f"  padding positions: {padding_mask.sum().item()}")
    
    # Forward through transformer
    hidden_states = model.transformer(embeddings, src_key_padding_mask=padding_mask)
    
    print_subheader("Transformer Output")
    print(f"  hidden_states shape: {hidden_states.shape}")
    print_array_stats("hidden_states", hidden_states.cpu())
    
    # Show change in representation
    print(f"\n  Representation change for first particle:")
    input_norm = embeddings[0, 0].norm().item()
    output_norm = hidden_states[0, 0].norm().item()
    print(f"    Input L2 norm: {input_norm:.4f}")
    print(f"    Output L2 norm: {output_norm:.4f}")
    
    return hidden_states


def trace_prediction_heads(model: ParticleTransformer, hidden_states: torch.Tensor) -> dict:
    """Trace the prediction heads."""
    print_header("STEP 10: Prediction Heads")
    
    print_subheader("Prediction Head Architectures")
    print(f"  pt_head: {model.pt_head}")
    print(f"  eta_head: {model.eta_head}")
    print(f"  phi_head: {model.phi_head}")
    print(f"  particle_id_head: {model.particle_id_head}")
    
    print_subheader("Computing Predictions")
    
    pt_pred = model.pt_head(hidden_states).squeeze(-1)
    eta_pred = model.eta_head(hidden_states).squeeze(-1)
    phi_pred = model.phi_head(hidden_states).squeeze(-1)
    particle_id_pred = model.particle_id_head(hidden_states)
    
    predictions = {
        'pt': pt_pred,
        'eta': eta_pred,
        'phi': phi_pred,
        'particle_id': particle_id_pred,
        'hidden_states': hidden_states,
    }
    
    print_subheader("Prediction Outputs")
    print(f"  pt_pred shape: {pt_pred.shape}")
    print_array_stats("pt_pred", pt_pred.cpu())
    
    print(f"\n  eta_pred shape: {eta_pred.shape}")
    print_array_stats("eta_pred", eta_pred.cpu())
    
    print(f"\n  phi_pred shape: {phi_pred.shape}")
    print_array_stats("phi_pred", phi_pred.cpu())
    
    print(f"\n  particle_id_pred shape: {particle_id_pred.shape}")
    print(f"  particle_id_pred: (batch, seq, n_classes={particle_id_pred.shape[-1]})")
    
    # Show predicted class probabilities for first masked particle
    probs = torch.softmax(particle_id_pred[0, 0], dim=-1)
    top_k = torch.topk(probs, k=3)
    print(f"\n  First particle - top 3 predicted classes:")
    for i, (prob, cls) in enumerate(zip(top_k.values, top_k.indices)):
        print(f"    {i+1}. class {cls.item()}: {prob.item():.4f}")
    
    return predictions


def trace_loss_computation(model: ParticleTransformer, predictions: dict, mask_targets: dict) -> dict:
    """Trace the loss computation."""
    print_header("STEP 11: Loss Computation")
    
    prediction_mask = mask_targets['mask']
    n_masked = prediction_mask.sum().item()
    
    print_subheader("Loss Computation Details")
    print(f"  Number of masked positions: {n_masked}")
    print(f"  Loss computed only on masked positions: True")
    
    # Handle empty mask case
    if n_masked == 0:
        print("\n  WARNING: No masked positions! Skipping detailed target/prediction analysis.")
        print("  Loss will be zero and no gradients will be computed.")
        losses = model.compute_loss(predictions, mask_targets)
        print_subheader("Loss Values")
        for name, value in losses.items():
            print(f"  {name}: {value.item():.6f}")
        return losses
    
    print_subheader("Target Values at Masked Positions")
    masked_pt = mask_targets['pt'][prediction_mask]
    masked_eta = mask_targets['eta'][prediction_mask]
    masked_phi = mask_targets['phi'][prediction_mask]
    masked_pid = mask_targets['particle_id'][prediction_mask]
    
    print_array_stats("target_pt", masked_pt.cpu())
    print_array_stats("target_eta", masked_eta.cpu())
    print_array_stats("target_phi", masked_phi.cpu())
    print_array_stats("target_particle_id", masked_pid.cpu())
    
    print_subheader("Predicted Values at Masked Positions")
    pred_pt = predictions['pt'][prediction_mask]
    pred_eta = predictions['eta'][prediction_mask]
    pred_phi = predictions['phi'][prediction_mask]
    pred_pid = predictions['particle_id'][prediction_mask]
    
    print_array_stats("pred_pt", pred_pt.cpu())
    print_array_stats("pred_eta", pred_eta.cpu())
    print_array_stats("pred_phi", pred_phi.cpu())
    print(f"  pred_particle_id: shape={pred_pid.shape}")
    
    # Compute losses
    losses = model.compute_loss(predictions, mask_targets)
    
    print_subheader("Loss Values")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
    
    print_subheader("Loss Breakdown")
    total = losses['total_loss'].item()
    if total > 0:
        print(f"  pt_loss contribution: {losses['pt_loss'].item() / total * 100:.1f}%")
        print(f"  eta_loss contribution: {losses['eta_loss'].item() / total * 100:.1f}%")
        print(f"  phi_loss contribution: {losses['phi_loss'].item() / total * 100:.1f}%")
        print(f"  particle_id_loss contribution: {losses['particle_id_loss'].item() / total * 100:.1f}%")
    
    return losses


def trace_backward_pass(losses: dict, model: ParticleTransformer) -> None:
    """Trace the backward pass."""
    print_header("STEP 12: Backward Pass (Gradient Computation)")
    
    total_loss = losses['total_loss']
    
    # Guard against zero loss or non-differentiable tensor
    if total_loss.item() == 0.0:
        print("  WARNING: Total loss is zero.")
        print("  Skipping backward pass - this event had no masked positions or all losses were zero.")
        return
    
    if not total_loss.requires_grad:
        print("  WARNING: Total loss tensor does not require gradients.")
        print("  Skipping backward pass - loss was computed without gradient tracking.")
        return
    
    print_subheader("Before Backward")
    # Check gradient status before
    grad_params = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {grad_params}/{total_params}")
    
    # Backward pass
    print("\nComputing gradients (loss.backward())...")
    total_loss.backward()
    
    print_subheader("After Backward")
    grad_params = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {grad_params}/{total_params}")
    
    # Show gradient statistics for key layers
    print_subheader("Gradient Statistics (Sample Layers)")
    
    # Determine embedding layer to check
    if hasattr(model.embedding, 'mlp'):
        emb_name = "embedding.mlp.0.weight"
        emb_param = model.embedding.mlp[0].weight
    else:
        emb_name = "embedding.pt_proj.weight"
        emb_param = model.embedding.pt_proj.weight
    
    layers_to_check = [
        (emb_name, emb_param),
        ("pt_head.weight", model.pt_head.weight),
        ("particle_id_head.weight", model.particle_id_head.weight),
    ]
    
    for name, param in layers_to_check:
        if param.grad is not None:
            grad = param.grad
            print(f"\n  {name}:")
            print(f"    shape: {grad.shape}")
            print(f"    grad norm: {grad.norm().item():.6f}")
            print(f"    grad mean: {grad.mean().item():.8f}")
            print(f"    grad std: {grad.std().item():.8f}")
            print(f"    grad min: {grad.min().item():.8f}")
            print(f"    grad max: {grad.max().item():.8f}")


def main():
    parser = argparse.ArgumentParser(description="Trace the complete training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/preprocessed_h5.yaml",
        help="Path to data config YAML file",
    )
    parser.add_argument(
        "--event-idx",
        type=int,
        default=0,
        help="Event index to trace",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(" PARTICLEMAN TRAINING PIPELINE TRACE")
    print(" Full visibility into every training step")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"Event Index: {args.event_idx}")
    print(f"Device: {args.device}")
    
    device = torch.device(args.device)
    
    # STEP 1: Config Loading
    config = trace_config_loading(args.config)
    
    # STEP 2: Raw Data Loading
    raw_event = trace_data_loading(config, args.event_idx)
    
    # STEP 3: Preprocessing
    processed_event, loader = trace_preprocessing(config, args.event_idx)
    
    # STEP 4: PyTorch Dataset
    sample = trace_dataset_conversion(config, args.event_idx)
    
    # STEP 5: Batching
    batch = trace_batching(sample)
    
    # Create model for remaining steps
    print_header("MODEL INITIALIZATION")
    model_config = ParticleConfig(
        max_particles=config.preprocessing.max_particles,
        n_particle_types=16,
        mask_prob=0.15,
        d_model=128,  # Smaller for demo
        n_heads=4,
        n_layers=2,
    )
    print(f"Creating ParticleTransformer with config:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  mask_prob: {model_config.mask_prob}")
    print(f"  embedding_type: {model_config.embedding_type}")
    print(f"  phi_encoding: {model_config.phi_encoding}")
    
    model = ParticleTransformer(model_config).to(device)
    model.train()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    # STEP 6: Masking
    masked_inputs, mask_targets, padding_mask = trace_masking(model, batch, device)
    
    # STEP 7: Normalization
    pt_norm, eta_norm, phi_processed = trace_normalization(model, masked_inputs)
    
    # STEP 8: Embedding
    embeddings = trace_embedding(
        model, pt_norm, eta_norm, phi_processed, 
        masked_inputs['particle_id']
    )
    
    # STEP 9: Transformer
    hidden_states = trace_transformer(model, embeddings, padding_mask)
    
    # STEP 10: Prediction Heads
    predictions = trace_prediction_heads(model, hidden_states)
    
    # STEP 11: Loss Computation
    losses = trace_loss_computation(model, predictions, mask_targets)
    
    # STEP 12: Backward Pass
    trace_backward_pass(losses, model)
    
    # Summary
    print_header("PIPELINE TRACE COMPLETE", char="*")
    print("All 12 steps of the training pipeline have been traced.")
    print("\nKey observations:")
    print(f"  - Input particles: {raw_event['n_particles']}")
    print(f"  - After preprocessing: {processed_event['n_particles']}")
    print(f"  - Masked positions: {mask_targets['mask'].sum().item()}")
    print(f"  - Total loss: {losses['total_loss'].item():.6f}")
    print(f"\nTo run actual training, use the ParticleTrainer class.")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
