#!/usr/bin/env python3
"""Post-processing compression script for Gaussian splat PLY files.

This script provides multiple compression strategies:
1. Quantization (float32 -> float16 or quantized integers)
2. Pruning (remove low-opacity Gaussians)
3. Delta encoding (for interframe compression of sequences)
4. General compression (gzip/lzma)

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import logging
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from sharp.utils.gaussians import load_ply, save_ply, Gaussians3D, SceneMetaData
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

CompressionMethod = Literal["none", "gzip", "lzma"]


@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters."""
    position_bits: int = 16  # 16 = float16, 32 = float32
    color_bits: int = 8      # 8 = uint8, 16 = float16, 32 = float32
    scale_bits: int = 16
    rotation_bits: int = 16
    opacity_bits: int = 8


@dataclass
class CompressionStats:
    """Statistics from compression operation."""
    original_size_bytes: int
    compressed_size_bytes: int
    original_gaussian_count: int
    compressed_gaussian_count: int
    compression_ratio: float
    
    def __str__(self) -> str:
        return (
            f"Original: {self.original_size_bytes / 1024 / 1024:.2f} MB, "
            f"{self.original_gaussian_count:,} Gaussians\n"
            f"Compressed: {self.compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"{self.compressed_gaussian_count:,} Gaussians\n"
            f"Ratio: {self.compression_ratio:.2f}x"
        )


# =============================================================================
# Quantization Functions
# =============================================================================

def quantize_tensor_float16(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to float16 precision."""
    return tensor.half().float()


def quantize_tensor_to_bits(
    tensor: torch.Tensor,
    bits: int,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> torch.Tensor:
    """Quantize tensor to fixed-point representation with given bit depth."""
    if min_val is None:
        min_val = tensor.min().item()
    if max_val is None:
        max_val = tensor.max().item()
    
    range_val = max_val - min_val
    if range_val < 1e-8:
        return tensor  # Avoid division by zero
    
    scale = (2**bits - 1) / range_val
    quantized = ((tensor - min_val) * scale).round().clamp(0, 2**bits - 1)
    return quantized / scale + min_val


def quantize_positions(pos: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize position vectors."""
    if bits == 32:
        return pos
    elif bits == 16:
        return quantize_tensor_float16(pos)
    else:
        # Quantize each dimension independently for better precision
        result = torch.zeros_like(pos)
        for i in range(pos.shape[-1]):
            result[..., i] = quantize_tensor_to_bits(pos[..., i], bits)
        return result


def quantize_colors(colors: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize color values (assumed to be in [0, 1] range)."""
    if bits == 32:
        return colors
    elif bits == 16:
        return quantize_tensor_float16(colors)
    elif bits == 8:
        # Standard 8-bit color quantization
        return (colors * 255).round().clamp(0, 255) / 255.0
    else:
        scale = 2**bits - 1
        return (colors * scale).round().clamp(0, scale) / scale


def quantize_scales(scales: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize scale values (using log-space for better distribution)."""
    if bits == 32:
        return scales
    elif bits == 16:
        return quantize_tensor_float16(scales)
    else:
        # Quantize in log space for better representation of small values
        log_scales = torch.log(scales + 1e-8)
        log_min, log_max = log_scales.min().item(), log_scales.max().item()
        quantized_log = quantize_tensor_to_bits(log_scales, bits, log_min, log_max)
        return torch.exp(quantized_log)


def quantize_quaternions(quats: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize quaternion values (normalized, in [-1, 1] range)."""
    if bits == 32:
        return quats
    elif bits == 16:
        return quantize_tensor_float16(quats)
    else:
        # Quaternions are typically normalized to unit length
        # Quantize in [-1, 1] range
        quantized = quantize_tensor_to_bits(quats, bits, -1.0, 1.0)
        # Re-normalize after quantization
        return quantized / (quantized.norm(dim=-1, keepdim=True) + 1e-8)


def quantize_opacities(opacities: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize opacity values (in [0, 1] range)."""
    if bits == 32:
        return opacities
    elif bits == 16:
        return quantize_tensor_float16(opacities)
    elif bits == 8:
        return (opacities * 255).round().clamp(0, 255) / 255.0
    else:
        scale = 2**bits - 1
        return (opacities * scale).round().clamp(0, scale) / scale


def quantize_gaussians(
    gaussians: Gaussians3D,
    config: QuantizationConfig,
) -> Gaussians3D:
    """Apply quantization to all Gaussian parameters."""
    LOGGER.info(
        f"Quantizing: pos={config.position_bits}bit, color={config.color_bits}bit, "
        f"scale={config.scale_bits}bit, rot={config.rotation_bits}bit, "
        f"opacity={config.opacity_bits}bit"
    )
    
    return Gaussians3D(
        mean_vectors=quantize_positions(gaussians.mean_vectors, config.position_bits),
        singular_values=quantize_scales(gaussians.singular_values, config.scale_bits),
        quaternions=quantize_quaternions(gaussians.quaternions, config.rotation_bits),
        colors=quantize_colors(gaussians.colors, config.color_bits),
        opacities=quantize_opacities(gaussians.opacities, config.opacity_bits),
    )


# =============================================================================
# Pruning Functions
# =============================================================================

def prune_gaussians(
    gaussians: Gaussians3D,
    opacity_threshold: float = 0.01,
    scale_threshold: Optional[float] = None,
) -> Gaussians3D:
    """Remove Gaussians below opacity and/or scale thresholds."""
    opacities = gaussians.opacities.flatten()
    mask = opacities >= opacity_threshold
    
    if scale_threshold is not None:
        # Also filter by scale (remove very small Gaussians)
        max_scales = gaussians.singular_values.max(dim=-1)[0].flatten()
        mask = mask & (max_scales >= scale_threshold)
    
    num_kept = mask.sum().item()
    num_total = len(mask)
    
    if num_kept == 0:
        LOGGER.warning("All Gaussians would be pruned. Keeping original.")
        return gaussians
    
    LOGGER.info(
        f"Pruning: {num_kept:,}/{num_total:,} Gaussians kept "
        f"({100 * num_kept / num_total:.1f}%)"
    )
    
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=gaussians.opacities[:, mask],
    )


# =============================================================================
# Delta Encoding (Interframe Compression)
# =============================================================================

def match_gaussians_spatially(
    base: Gaussians3D,
    current: Gaussians3D,
    max_match_distance: float = 0.1,  # Maximum distance for matching
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match Gaussians between frames using spatial proximity.
    
    Args:
        base: Base frame Gaussians
        current: Current frame Gaussians
        max_match_distance: Maximum 3D distance for matching (in same units as positions)
        
    Returns:
        Tuple of (base_indices, current_indices, unmatched_current_indices)
        - base_indices: Indices in base that have matches
        - current_indices: Corresponding indices in current
        - unmatched_current_indices: Indices in current with no match (new Gaussians)
    """
    # Extract positions
    base_pos = base.mean_vectors[0].cpu().numpy()  # Shape: [N, 3]
    current_pos = current.mean_vectors[0].cpu().numpy()  # Shape: [M, 3]
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(base_pos)
    
    # Find nearest base Gaussian for each current Gaussian
    distances, base_indices = tree.query(current_pos, k=1)
    
    # Filter matches by distance threshold
    valid_mask = distances <= max_match_distance
    matched_base_indices = base_indices[valid_mask]
    matched_current_indices = torch.where(torch.from_numpy(valid_mask))[0]
    
    # Find unmatched current Gaussians (new ones)
    unmatched_mask = ~valid_mask
    unmatched_current_indices = torch.where(torch.from_numpy(unmatched_mask))[0]
    
    return (
        torch.from_numpy(matched_base_indices),
        matched_current_indices,
        unmatched_current_indices,
    )


@dataclass
class DeltaFrame:
    """Represents a delta-encoded frame."""
    frame_index: int
    base_frame_index: int
    delta_mean: torch.Tensor
    delta_scales: torch.Tensor
    delta_quaternions: torch.Tensor
    delta_colors: torch.Tensor
    delta_opacities: torch.Tensor
    # Sparse encoding: indices of changed Gaussians (None = all changed)
    changed_indices: Optional[torch.Tensor] = None
    # Spatial matching: indices for matching between frames
    base_match_indices: Optional[torch.Tensor] = None
    current_match_indices: Optional[torch.Tensor] = None
    is_new_gaussian: Optional[torch.Tensor] = None  # Boolean mask: True = new, False = matched
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        result = {
            "frame_index": self.frame_index,
            "base_frame_index": self.base_frame_index,
            "delta_mean": self.delta_mean.cpu().numpy().tolist(),
            "delta_scales": self.delta_scales.cpu().numpy().tolist(),
            "delta_quaternions": self.delta_quaternions.cpu().numpy().tolist(),
            "delta_colors": self.delta_colors.cpu().numpy().tolist(),
            "delta_opacities": self.delta_opacities.cpu().numpy().tolist(),
        }
        if self.changed_indices is not None:
            result["changed_indices"] = self.changed_indices.cpu().numpy().tolist()
        if self.base_match_indices is not None:
            result["base_match_indices"] = self.base_match_indices.cpu().numpy().tolist()
        if self.current_match_indices is not None:
            result["current_match_indices"] = self.current_match_indices.cpu().numpy().tolist()
        if self.is_new_gaussian is not None:
            result["is_new_gaussian"] = self.is_new_gaussian.cpu().numpy().tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "DeltaFrame":
        """Construct from dictionary."""
        changed_indices = None
        if "changed_indices" in data:
            changed_indices = torch.tensor(data["changed_indices"])
        
        base_match_indices = None
        if "base_match_indices" in data:
            base_match_indices = torch.tensor(data["base_match_indices"])
        
        current_match_indices = None
        if "current_match_indices" in data:
            current_match_indices = torch.tensor(data["current_match_indices"])
        
        is_new_gaussian = None
        if "is_new_gaussian" in data:
            is_new_gaussian = torch.tensor(data["is_new_gaussian"], dtype=torch.bool)
        
        return cls(
            frame_index=data["frame_index"],
            base_frame_index=data["base_frame_index"],
            delta_mean=torch.tensor(data["delta_mean"]),
            delta_scales=torch.tensor(data["delta_scales"]),
            delta_quaternions=torch.tensor(data["delta_quaternions"]),
            delta_colors=torch.tensor(data["delta_colors"]),
            delta_opacities=torch.tensor(data["delta_opacities"]),
            changed_indices=changed_indices,
            base_match_indices=base_match_indices,
            current_match_indices=current_match_indices,
            is_new_gaussian=is_new_gaussian,
        )


def compute_delta(
    base: Gaussians3D,
    current: Gaussians3D,
    sparse_threshold: float = 1e-4,
    use_sparse: bool = True,
    match_spatially: bool = True,
    max_match_distance: float = 0.1,
    quantize_deltas: bool = True,
    delta_quantization_bits: int = 16,
    adaptive_threshold: bool = True,
    threshold_percentile: float = 50.0,
) -> tuple[Gaussians3D, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute delta between two Gaussian sets with optional spatial matching.
    
    Args:
        base: Base frame Gaussians
        current: Current frame Gaussians  
        sparse_threshold: Threshold below which changes are considered zero
        use_sparse: If True, only store changed Gaussians
        match_spatially: If True, match Gaussians by spatial proximity instead of index
        max_match_distance: Maximum distance for spatial matching
        
    Returns:
        Tuple of (delta Gaussians, changed indices, base_match_indices, current_match_indices, is_new_gaussian)
        - If match_spatially=True, returns matching information
        - If match_spatially=False, base_match_indices, current_match_indices, and is_new_gaussian are None
    """
    if base.mean_vectors.shape != current.mean_vectors.shape and not match_spatially:
        raise ValueError(
            f"Shape mismatch: base {base.mean_vectors.shape} vs "
            f"current {current.mean_vectors.shape}. Use match_spatially=True for different counts."
        )
    
    if match_spatially:
        # Match Gaussians spatially
        base_match_idx, current_match_idx, unmatched_current_idx = match_gaussians_spatially(
            base, current, max_match_distance
        )
        
        num_matched = len(base_match_idx)
        num_unmatched = len(unmatched_current_idx)
        num_total_current = current.mean_vectors.shape[1]
        
        LOGGER.info(
            f"Spatial matching: {num_matched:,}/{num_total_current:,} matched, "
            f"{num_unmatched:,} new Gaussians"
        )
        
        # Compute deltas only for matched pairs
        if num_matched > 0:
            base_matched = Gaussians3D(
                mean_vectors=base.mean_vectors[:, base_match_idx],
                singular_values=base.singular_values[:, base_match_idx],
                quaternions=base.quaternions[:, base_match_idx],
                colors=base.colors[:, base_match_idx],
                opacities=base.opacities[:, base_match_idx],
            )
            
            current_matched = Gaussians3D(
                mean_vectors=current.mean_vectors[:, current_match_idx],
                singular_values=current.singular_values[:, current_match_idx],
                quaternions=current.quaternions[:, current_match_idx],
                colors=current.colors[:, current_match_idx],
                opacities=current.opacities[:, current_match_idx],
            )
            
            delta_mean = current_matched.mean_vectors - base_matched.mean_vectors
            delta_scales = current_matched.singular_values - base_matched.singular_values
            delta_quats = current_matched.quaternions - base_matched.quaternions
            delta_colors = current_matched.colors - base_matched.colors
            delta_opacities = current_matched.opacities - base_matched.opacities
            
            # For unmatched Gaussians, store them as "new" (full values, not deltas)
            if num_unmatched > 0:
                new_gaussians = Gaussians3D(
                    mean_vectors=current.mean_vectors[:, unmatched_current_idx],
                    singular_values=current.singular_values[:, unmatched_current_idx],
                    quaternions=current.quaternions[:, unmatched_current_idx],
                    colors=current.colors[:, unmatched_current_idx],
                    opacities=current.opacities[:, unmatched_current_idx],
                )
                # Concatenate deltas with new Gaussians
                delta_mean = torch.cat([delta_mean, new_gaussians.mean_vectors], dim=1)
                delta_scales = torch.cat([delta_scales, new_gaussians.singular_values], dim=1)
                delta_quats = torch.cat([delta_quats, new_gaussians.quaternions], dim=1)
                delta_colors = torch.cat([delta_colors, new_gaussians.colors], dim=1)
                delta_opacities = torch.cat([delta_opacities, new_gaussians.opacities], dim=1)
                
                # Create mask: False for matched (deltas), True for new (full values)
                device = delta_mean.device
                is_new = torch.cat([
                    torch.zeros(num_matched, dtype=torch.bool, device=device),
                    torch.ones(num_unmatched, dtype=torch.bool, device=device),
                ])
            else:
                device = delta_mean.device
                is_new = torch.zeros(num_matched, dtype=torch.bool, device=device)
        else:
            # No matches - all are new
            delta_mean = current.mean_vectors
            delta_scales = current.singular_values
            delta_quats = current.quaternions
            delta_colors = current.colors
            delta_opacities = current.opacities
            base_match_idx = None
            current_match_idx = None
            device = current.mean_vectors.device
            is_new = torch.ones(num_total_current, dtype=torch.bool, device=device)
        
        changed_indices = None
        if use_sparse and num_matched > 0:
            # IMPROVED: Use relative change magnitude (normalized by base values)
            # This is more robust than absolute thresholds
            base_matched = Gaussians3D(
                mean_vectors=base.mean_vectors[:, base_match_idx],
                singular_values=base.singular_values[:, base_match_idx],
                quaternions=base.quaternions[:, base_match_idx],
                colors=base.colors[:, base_match_idx],
                opacities=base.opacities[:, base_match_idx],
            )
            
            # Relative change: delta / (base + epsilon)
            eps = 1e-8
            rel_delta_mean = (delta_mean[:, :num_matched].abs() / (base_matched.mean_vectors.abs() + eps)).max(dim=-1)[0]
            rel_delta_scales = (delta_scales[:, :num_matched].abs() / (base_matched.singular_values.abs() + eps)).max(dim=-1)[0]
            rel_delta_colors = (delta_colors[:, :num_matched].abs() / (base_matched.colors.abs() + eps)).max(dim=-1)[0]
            rel_delta_opacities = (delta_opacities[:, :num_matched].abs() / (base_matched.opacities.abs() + eps))
            
            # For quaternions, use absolute change (they're normalized)
            abs_delta_quats = delta_quats[:, :num_matched].abs().max(dim=-1)[0]
            
            # Combined relative change magnitude
            change_magnitude = (
                rel_delta_mean +
                rel_delta_scales +
                abs_delta_quats * 0.1 +  # Weight quaternions less
                rel_delta_colors +
                rel_delta_opacities
            ).flatten()
            
            # ADAPTIVE THRESHOLD: Use percentile instead of fixed threshold
            if adaptive_threshold:
                threshold_value = torch.quantile(change_magnitude, threshold_percentile / 100.0).item()
                actual_threshold = max(threshold_value, sparse_threshold)  # Don't go below minimum
                LOGGER.debug(f"Adaptive threshold: {actual_threshold:.2e} (percentile {threshold_percentile}%)")
            else:
                actual_threshold = sparse_threshold
            
            changed_mask = change_magnitude > actual_threshold
            changed_indices = torch.where(changed_mask)[0]
            
            num_changed = len(changed_indices)
            # Log statistics about delta magnitudes
            max_delta = change_magnitude.max().item()
            mean_delta = change_magnitude.mean().item()
            median_delta = change_magnitude.median().item()
            
            LOGGER.info(
                f"Delta encoding: {num_changed:,}/{num_matched:,} matched Gaussians changed "
                f"({100 * num_changed / num_matched:.1f}%) | "
                f"Threshold: {actual_threshold:.2e} | "
                f"Max: {max_delta:.2e} | Mean: {mean_delta:.2e} | Median: {median_delta:.2e}"
            )
            
            # Only use sparse if <90% changed
            if num_changed < num_matched * 0.9:
                # Keep changed matched + all new Gaussians
                if num_unmatched > 0:
                    all_indices = torch.cat([
                        changed_indices,
                        torch.arange(num_matched, num_matched + num_unmatched, device=changed_indices.device)
                    ])
                else:
                    all_indices = changed_indices
                
                delta_mean = delta_mean[:, all_indices]
                delta_scales = delta_scales[:, all_indices]
                delta_quats = delta_quats[:, all_indices]
                delta_colors = delta_colors[:, all_indices]
                delta_opacities = delta_opacities[:, all_indices]
                
                # Update indices to reflect sparse encoding
                if base_match_idx is not None:
                    base_match_idx = base_match_idx[changed_indices]
                if current_match_idx is not None:
                    current_match_idx = current_match_idx[changed_indices]
                is_new = is_new[all_indices]
                
                # QUANTIZE DELTAS: Reduce precision of stored deltas
                if quantize_deltas:
                    if delta_quantization_bits == 16:
                        # Float16 quantization
                        delta_mean = delta_mean.half().float()
                        delta_scales = delta_scales.half().float()
                        delta_quats = delta_quats.half().float()
                        delta_colors = delta_colors.half().float()
                        delta_opacities = delta_opacities.half().float()
                        LOGGER.debug("Quantized deltas to float16")
            else:
                changed_indices = None  # Use dense encoding
                
                # Still quantize even if dense
                if quantize_deltas and delta_quantization_bits == 16:
                    delta_mean = delta_mean.half().float()
                    delta_scales = delta_scales.half().float()
                    delta_quats = delta_quats.half().float()
                    delta_colors = delta_colors.half().float()
                    delta_opacities = delta_opacities.half().float()
        
    else:
        # Original index-based matching (assumes same order)
        delta_mean = current.mean_vectors - base.mean_vectors
        delta_scales = current.singular_values - base.singular_values
        delta_quats = current.quaternions - base.quaternions
        delta_colors = current.colors - base.colors
        delta_opacities = current.opacities - base.opacities
        
        base_match_idx = None
        current_match_idx = None
        is_new = None
        
        changed_indices = None
        if use_sparse:
            change_magnitude = (
                delta_mean.abs().max(dim=-1)[0] +
                delta_scales.abs().max(dim=-1)[0] +
                delta_quats.abs().max(dim=-1)[0] +
                delta_colors.abs().max(dim=-1)[0] +
                delta_opacities.abs()
            ).flatten()
            
            changed_mask = change_magnitude > sparse_threshold
            changed_indices = torch.where(changed_mask)[0]
            
            num_changed = len(changed_indices)
            num_total = len(change_magnitude)
            
            LOGGER.info(
                f"Delta encoding: {num_changed:,}/{num_total:,} Gaussians changed "
                f"({100 * num_changed / num_total:.1f}%)"
            )
            
            if num_changed < num_total * 0.9:
                delta_mean = delta_mean[:, changed_indices]
                delta_scales = delta_scales[:, changed_indices]
                delta_quats = delta_quats[:, changed_indices]
                delta_colors = delta_colors[:, changed_indices]
                delta_opacities = delta_opacities[:, changed_indices]
            else:
                changed_indices = None
    
    delta_gaussians = Gaussians3D(
        mean_vectors=delta_mean,
        singular_values=delta_scales,
        quaternions=delta_quats,
        colors=delta_colors,
        opacities=delta_opacities,
    )
    
    return delta_gaussians, changed_indices, base_match_idx, current_match_idx, is_new


def apply_delta(
    base: Gaussians3D,
    delta: Gaussians3D,
    changed_indices: Optional[torch.Tensor] = None,
    base_match_indices: Optional[torch.Tensor] = None,
    current_match_indices: Optional[torch.Tensor] = None,
    is_new_gaussian: Optional[torch.Tensor] = None,
) -> Gaussians3D:
    """
    Apply delta to reconstruct current frame from base.
    
    Args:
        base: Base frame Gaussians
        delta: Delta Gaussians
        changed_indices: Indices of changed Gaussians (for sparse encoding)
        base_match_indices: Indices in base that were matched (for spatial matching)
        current_match_indices: Indices in current that were matched (for spatial matching)
        is_new_gaussian: Boolean mask indicating which deltas are new Gaussians vs matched deltas
    """
    if base_match_indices is not None and current_match_indices is not None:
        # Spatial matching mode
        num_deltas = delta.mean_vectors.shape[1]
        
        if is_new_gaussian is not None:
            # Separate matched deltas from new Gaussians
            matched_mask = ~is_new_gaussian
            new_mask = is_new_gaussian
            
            matched_deltas = Gaussians3D(
                mean_vectors=delta.mean_vectors[:, matched_mask],
                singular_values=delta.singular_values[:, matched_mask],
                quaternions=delta.quaternions[:, matched_mask],
                colors=delta.colors[:, matched_mask],
                opacities=delta.opacities[:, matched_mask],
            )
            
            new_gaussians = Gaussians3D(
                mean_vectors=delta.mean_vectors[:, new_mask],
                singular_values=delta.singular_values[:, new_mask],
                quaternions=delta.quaternions[:, new_mask],
                colors=delta.colors[:, new_mask],
                opacities=delta.opacities[:, new_mask],
            )
            
            # Apply deltas to matched Gaussians
            matched_base = Gaussians3D(
                mean_vectors=base.mean_vectors[:, base_match_indices],
                singular_values=base.singular_values[:, base_match_indices],
                quaternions=base.quaternions[:, base_match_indices],
                colors=base.colors[:, base_match_indices],
                opacities=base.opacities[:, base_match_indices],
            )
            
            matched_result = Gaussians3D(
                mean_vectors=matched_base.mean_vectors + matched_deltas.mean_vectors,
                singular_values=matched_base.singular_values + matched_deltas.singular_values,
                quaternions=matched_base.quaternions + matched_deltas.quaternions,
                colors=matched_base.colors + matched_deltas.colors,
                opacities=matched_base.opacities + matched_deltas.opacities,
            )
            
            # Combine matched results with new Gaussians
            mean_vectors = torch.cat([matched_result.mean_vectors, new_gaussians.mean_vectors], dim=1)
            singular_values = torch.cat([matched_result.singular_values, new_gaussians.singular_values], dim=1)
            quaternions = torch.cat([matched_result.quaternions, new_gaussians.quaternions], dim=1)
            colors = torch.cat([matched_result.colors, new_gaussians.colors], dim=1)
            opacities = torch.cat([matched_result.opacities, new_gaussians.opacities], dim=1)
        else:
            # All are matched deltas
            matched_base = Gaussians3D(
                mean_vectors=base.mean_vectors[:, base_match_indices],
                singular_values=base.singular_values[:, base_match_indices],
                quaternions=base.quaternions[:, base_match_indices],
                colors=base.colors[:, base_match_indices],
                opacities=base.opacities[:, base_match_indices],
            )
            
            mean_vectors = matched_base.mean_vectors + delta.mean_vectors
            singular_values = matched_base.singular_values + delta.singular_values
            quaternions = matched_base.quaternions + delta.quaternions
            colors = matched_base.colors + delta.colors
            opacities = matched_base.opacities + delta.opacities
        
        return Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=singular_values,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
        )
    
    elif changed_indices is not None:
        # Sparse delta - only update changed indices (index-based matching)
        mean_vectors = base.mean_vectors.clone()
        singular_values = base.singular_values.clone()
        quaternions = base.quaternions.clone()
        colors = base.colors.clone()
        opacities = base.opacities.clone()
        
        mean_vectors[:, changed_indices] += delta.mean_vectors
        singular_values[:, changed_indices] += delta.singular_values
        quaternions[:, changed_indices] += delta.quaternions
        colors[:, changed_indices] += delta.colors
        opacities[:, changed_indices] += delta.opacities
    else:
        # Dense delta - apply to all (index-based matching)
        mean_vectors = base.mean_vectors + delta.mean_vectors
        singular_values = base.singular_values + delta.singular_values
        quaternions = base.quaternions + delta.quaternions
        colors = base.colors + delta.colors
        opacities = base.opacities + delta.opacities
    
    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


# =============================================================================
# File I/O with Compression
# =============================================================================

def save_compressed_ply(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    path: Path,
    compression: CompressionMethod = "none",
) -> int:
    """Save PLY with optional compression, returns file size in bytes."""
    if compression == "none":
        save_ply(gaussians, metadata.focal_length_px, metadata.resolution_px[::-1], path)
        return path.stat().st_size
    
    # Save to temporary uncompressed file first
    temp_path = path.with_suffix('.ply.tmp')
    save_ply(gaussians, metadata.focal_length_px, metadata.resolution_px[::-1], temp_path)
    
    # Read and compress
    with open(temp_path, 'rb') as f_in:
        data = f_in.read()
    
    if compression == "gzip":
        compressed_path = path.with_suffix('.ply.gz')
        with gzip.open(compressed_path, 'wb', compresslevel=9) as f_out:
            f_out.write(data)
    elif compression == "lzma":
        compressed_path = path.with_suffix('.ply.xz')
        with lzma.open(compressed_path, 'wb', preset=9) as f_out:
            f_out.write(data)
    
    temp_path.unlink()
    return compressed_path.stat().st_size


def load_compressed_ply(path: Path) -> tuple[Gaussians3D, SceneMetaData]:
    """Load PLY with automatic decompression detection."""
    suffix = path.suffix.lower()
    
    if suffix == '.gz':
        temp_path = path.with_suffix('')
        with gzip.open(path, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                f_out.write(f_in.read())
        result = load_ply(temp_path)
        temp_path.unlink()
        return result
    elif suffix == '.xz':
        temp_path = path.with_suffix('')
        with lzma.open(path, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                f_out.write(f_in.read())
        result = load_ply(temp_path)
        temp_path.unlink()
        return result
    else:
        return load_ply(path)


def save_delta_sequence(
    base_gaussians: Gaussians3D,
    base_metadata: SceneMetaData,
    deltas: list[tuple[DeltaFrame, SceneMetaData]],
    output_dir: Path,
    compression: CompressionMethod = "gzip",
) -> None:
    """Save a delta-encoded sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save base frame
    base_path = output_dir / "base.ply"
    save_compressed_ply(base_gaussians, base_metadata, base_path, compression)
    LOGGER.info(f"Saved base frame to {base_path}")
    
    # Save deltas as compressed JSON + binary
    deltas_data = []
    for delta_frame, metadata in deltas:
        delta_dict = delta_frame.to_dict()
        delta_dict["metadata"] = {
            "focal_length_px": metadata.focal_length_px,
            "resolution_px": list(metadata.resolution_px),
            "color_space": metadata.color_space,
        }
        deltas_data.append(delta_dict)
    
    deltas_path = output_dir / "deltas.json"
    if compression == "gzip":
        deltas_path = deltas_path.with_suffix('.json.gz')
        with gzip.open(deltas_path, 'wt') as f:
            json.dump(deltas_data, f)
    elif compression == "lzma":
        deltas_path = deltas_path.with_suffix('.json.xz')
        with lzma.open(deltas_path, 'wt') as f:
            json.dump(deltas_data, f)
    else:
        with open(deltas_path, 'w') as f:
            json.dump(deltas_data, f)
    
    LOGGER.info(f"Saved {len(deltas)} delta frames to {deltas_path}")


def load_delta_sequence(
    input_dir: Path,
) -> tuple[Gaussians3D, SceneMetaData, list[tuple[DeltaFrame, SceneMetaData]]]:
    """Load a delta-encoded sequence."""
    # Find and load base frame
    base_candidates = list(input_dir.glob("base.ply*"))
    if not base_candidates:
        raise FileNotFoundError(f"No base.ply found in {input_dir}")
    
    base_gaussians, base_metadata = load_compressed_ply(base_candidates[0])
    
    # Find and load deltas (try binary format first, fallback to JSON for compatibility)
    delta_candidates = list(input_dir.glob("deltas.pkl*")) + list(input_dir.glob("deltas.json*"))
    if not delta_candidates:
        return base_gaussians, base_metadata, []
    
    delta_path = delta_candidates[0]
    
    # Try binary format first
    if delta_path.suffix in ['.pkl', '.gz', '.xz'] and 'pkl' in delta_path.stem:
        if delta_path.suffix == '.gz':
            with gzip.open(delta_path, 'rb') as f:
                deltas = pickle.load(f)
        elif delta_path.suffix == '.xz':
            with lzma.open(delta_path, 'rb') as f:
                deltas = pickle.load(f)
        else:
            with open(delta_path, 'rb') as f:
                deltas = pickle.load(f)
    else:
        # Fallback to JSON format (legacy)
        if delta_path.suffix == '.gz':
            with gzip.open(delta_path, 'rt') as f:
                deltas_data = json.load(f)
        elif delta_path.suffix == '.xz':
            with lzma.open(delta_path, 'rt') as f:
                deltas_data = json.load(f)
        else:
            with open(delta_path, 'r') as f:
                deltas_data = json.load(f)
        
        deltas = []
        for delta_dict in deltas_data:
            meta_dict = delta_dict.pop("metadata")
            delta_frame = DeltaFrame.from_dict(delta_dict)
            metadata = SceneMetaData(
                focal_length_px=meta_dict["focal_length_px"],
                resolution_px=tuple(meta_dict["resolution_px"]),
                color_space=meta_dict["color_space"],
            )
            deltas.append((delta_frame, metadata))
    
    return base_gaussians, base_metadata, deltas


# =============================================================================
# Main Compression Functions
# =============================================================================

def compress_single_file(
    input_path: Path,
    output_path: Path,
    quantize: bool = True,
    quant_config: Optional[QuantizationConfig] = None,
    prune: bool = False,
    opacity_threshold: float = 0.01,
    scale_threshold: Optional[float] = None,
    compression: CompressionMethod = "none",
) -> CompressionStats:
    """Compress a single PLY file."""
    LOGGER.info(f"Loading {input_path}...")
    gaussians, metadata = load_ply(input_path)
    
    original_size = input_path.stat().st_size
    original_count = gaussians.mean_vectors.shape[1]
    
    LOGGER.info(f"Original: {original_size / 1024 / 1024:.2f} MB, {original_count:,} Gaussians")
    
    # Apply pruning first (reduces count)
    if prune:
        gaussians = prune_gaussians(
            gaussians,
            opacity_threshold=opacity_threshold,
            scale_threshold=scale_threshold,
        )
    
    # Apply quantization
    if quantize:
        config = quant_config or QuantizationConfig()
        gaussians = quantize_gaussians(gaussians, config)
    
    # Save with optional compression
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compressed_size = save_compressed_ply(gaussians, metadata, output_path, compression)
    compressed_count = gaussians.mean_vectors.shape[1]
    
    # Get actual output path (may have compression suffix)
    if compression == "gzip":
        actual_output = output_path.with_suffix('.ply.gz')
    elif compression == "lzma":
        actual_output = output_path.with_suffix('.ply.xz')
    else:
        actual_output = output_path
    
    LOGGER.info(f"Saved compressed file to {actual_output}")
    
    stats = CompressionStats(
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
        original_gaussian_count=original_count,
        compressed_gaussian_count=compressed_count,
        compression_ratio=original_size / compressed_size,
    )
    
    LOGGER.info(f"\n{stats}")
    return stats


def compress_sequence(
    input_dir: Path,
    output_dir: Path,
    use_delta: bool = False,
    quantize: bool = True,
    quant_config: Optional[QuantizationConfig] = None,
    prune: bool = False,
    opacity_threshold: float = 0.01,
    scale_threshold: Optional[float] = None,
    compression: CompressionMethod = "gzip",
    sparse_delta_threshold: float = 1e-4,
    keyframe_interval: int = 10,  # Insert keyframe every N frames
    max_match_distance: float = 0.1,  # Maximum distance for spatial matching
    quantize_deltas: bool = True,
    adaptive_threshold: bool = True,
    threshold_percentile: float = 50.0,
) -> list[CompressionStats]:
    """
    Compress a sequence of PLY files.
    
    Args:
        input_dir: Directory containing PLY files
        output_dir: Output directory
        use_delta: Use delta encoding between frames
        quantize: Apply quantization
        quant_config: Quantization configuration
        prune: Remove low-opacity Gaussians
        opacity_threshold: Threshold for pruning
        scale_threshold: Scale threshold for pruning
        compression: Compression method (none, gzip, lzma)
        sparse_delta_threshold: Threshold for sparse delta encoding
        keyframe_interval: Insert full keyframe every N frames for delta encoding
        max_match_distance: Maximum 3D distance for spatial matching between frames
    """
    ply_files = sorted(input_dir.glob("*.ply"))
    
    if not ply_files:
        LOGGER.error(f"No PLY files found in {input_dir}")
        return []
    
    LOGGER.info(f"Found {len(ply_files)} PLY files")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_list = []
    config = quant_config or QuantizationConfig()
    
    if use_delta:
        # Delta encoding mode
        LOGGER.info("Using delta encoding for interframe compression")
        
        base_gaussians = None
        base_metadata = None
        deltas = []
        total_original_size = 0
        total_compressed_size = 0
        
        for i, ply_file in enumerate(ply_files):
            LOGGER.info(f"\nProcessing {ply_file.name} ({i+1}/{len(ply_files)})...")
            
            gaussians, metadata = load_ply(ply_file)
            original_size = ply_file.stat().st_size
            total_original_size += original_size
            original_count = gaussians.mean_vectors.shape[1]
            
            # Apply pruning
            if prune:
                gaussians = prune_gaussians(
                    gaussians,
                    opacity_threshold=opacity_threshold,
                    scale_threshold=scale_threshold,
                )
            
            # Apply quantization
            if quantize:
                gaussians = quantize_gaussians(gaussians, config)
            
            is_keyframe = (i == 0) or (i % keyframe_interval == 0)
            
            if is_keyframe or base_gaussians is None:
                # Save as keyframe
                keyframe_path = output_dir / f"keyframe_{i:04d}.ply"
                compressed_size = save_compressed_ply(
                    gaussians, metadata, keyframe_path, compression
                )
                total_compressed_size += compressed_size
                
                base_gaussians = gaussians
                base_metadata = metadata
                base_index = i
                
                LOGGER.info(f"Saved keyframe {i}")
            else:
                # Compute and store delta with spatial matching
                delta_gaussians, changed_indices, base_match_idx, current_match_idx, is_new = compute_delta(
                    base_gaussians,
                    gaussians,
                    sparse_threshold=sparse_delta_threshold,
                    match_spatially=True,
                    max_match_distance=max_match_distance,
                    quantize_deltas=quantize_deltas,
                    delta_quantization_bits=16,
                    adaptive_threshold=adaptive_threshold,
                    threshold_percentile=threshold_percentile,
                )
                
                delta_frame = DeltaFrame(
                    frame_index=i,
                    base_frame_index=base_index,
                    delta_mean=delta_gaussians.mean_vectors,
                    delta_scales=delta_gaussians.singular_values,
                    delta_quaternions=delta_gaussians.quaternions,
                    delta_colors=delta_gaussians.colors,
                    delta_opacities=delta_gaussians.opacities,
                    changed_indices=changed_indices,
                    base_match_indices=base_match_idx,
                    current_match_indices=current_match_idx,
                    is_new_gaussian=is_new,
                )
                deltas.append((delta_frame, metadata))
            
            stats_list.append(CompressionStats(
                original_size_bytes=original_size,
                compressed_size_bytes=0,  # Will update after saving deltas
                original_gaussian_count=original_count,
                compressed_gaussian_count=gaussians.mean_vectors.shape[1],
                compression_ratio=0,
            ))
        
        # Save accumulated deltas (use numpy compressed format for better compression)
        if deltas:
            deltas_path = output_dir / "deltas.npz"
            
            # Build numpy arrays for all deltas
            delta_dict = {}
            for i, (delta_frame, metadata) in enumerate(deltas):
                prefix = f"frame_{i}"
                delta_dict[f"{prefix}_mean"] = delta_frame.delta_mean.cpu().numpy()
                delta_dict[f"{prefix}_scales"] = delta_frame.delta_scales.cpu().numpy()
                delta_dict[f"{prefix}_quats"] = delta_frame.delta_quaternions.cpu().numpy()
                delta_dict[f"{prefix}_colors"] = delta_frame.delta_colors.cpu().numpy()
                delta_dict[f"{prefix}_opacities"] = delta_frame.delta_opacities.cpu().numpy()
                
                if delta_frame.changed_indices is not None:
                    delta_dict[f"{prefix}_changed"] = delta_frame.changed_indices.cpu().numpy()
                if delta_frame.base_match_indices is not None:
                    delta_dict[f"{prefix}_base_match"] = delta_frame.base_match_indices.cpu().numpy()
                if delta_frame.current_match_indices is not None:
                    delta_dict[f"{prefix}_current_match"] = delta_frame.current_match_indices.cpu().numpy()
                if delta_frame.is_new_gaussian is not None:
                    delta_dict[f"{prefix}_is_new"] = delta_frame.is_new_gaussian.cpu().numpy()
                
                # Store metadata
                delta_dict[f"{prefix}_frame_idx"] = np.array([delta_frame.frame_index])
                delta_dict[f"{prefix}_base_idx"] = np.array([delta_frame.base_frame_index])
                delta_dict[f"{prefix}_metadata"] = np.array([
                    metadata.focal_length_px,
                    metadata.resolution_px[0],
                    metadata.resolution_px[1],
                ])
            
            # Save as compressed numpy format
            np.savez_compressed(deltas_path, **delta_dict)
            
            total_compressed_size += deltas_path.stat().st_size
            LOGGER.info(f"Saved {len(deltas)} delta frames to {deltas_path} ({deltas_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        overall_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
        LOGGER.info(
            f"\nSequence compression complete:\n"
            f"  Total original: {total_original_size / 1024 / 1024:.2f} MB\n"
            f"  Total compressed: {total_compressed_size / 1024 / 1024:.2f} MB\n"
            f"  Overall ratio: {overall_ratio:.2f}x"
        )
    
    else:
        # Regular per-file compression
        for i, ply_file in enumerate(ply_files):
            LOGGER.info(f"\nProcessing {ply_file.name} ({i+1}/{len(ply_files)})...")
            output_path = output_dir / ply_file.name
            
            stats = compress_single_file(
                ply_file,
                output_path,
                quantize=quantize,
                quant_config=config,
                prune=prune,
                opacity_threshold=opacity_threshold,
                scale_threshold=scale_threshold,
                compression=compression,
            )
            stats_list.append(stats)
        
        # Print summary
        if stats_list:
            total_original = sum(s.original_size_bytes for s in stats_list)
            total_compressed = sum(s.compressed_size_bytes for s in stats_list)
            avg_ratio = total_original / total_compressed if total_compressed > 0 else 0
            
            LOGGER.info(
                f"\nSequence compression complete:\n"
                f"  Total original: {total_original / 1024 / 1024:.2f} MB\n"
                f"  Total compressed: {total_compressed / 1024 / 1024:.2f} MB\n"
                f"  Average ratio: {avg_ratio:.2f}x"
            )
    
    return stats_list


def decompress_sequence(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Decompress a delta-encoded sequence back to individual PLY files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load keyframes
    keyframe_files = sorted(input_dir.glob("keyframe_*.ply*"))
    keyframes = {}
    
    for kf_path in keyframe_files:
        # Extract frame index from filename
        stem = kf_path.stem.replace('.ply', '')
        frame_idx = int(stem.split('_')[1])
        gaussians, metadata = load_compressed_ply(kf_path)
        keyframes[frame_idx] = (gaussians, metadata)
        
        # Save decompressed keyframe
        output_path = output_dir / f"{frame_idx:04d}.ply"
        save_ply(gaussians, metadata.focal_length_px, metadata.resolution_px[::-1], output_path)
        LOGGER.info(f"Decompressed keyframe {frame_idx}")
    
    # Load and apply deltas (try numpy format first, then pickle, then JSON)
    delta_files = list(input_dir.glob("deltas.npz")) + list(input_dir.glob("deltas.pkl*")) + list(input_dir.glob("deltas.json*"))
    if not delta_files:
        LOGGER.info("No delta files found, only keyframes decompressed")
        return
    
    delta_path = delta_files[0]
    
    # Try numpy compressed format first (most efficient)
    if delta_path.suffix == '.npz':
        npz_data = np.load(delta_path, allow_pickle=False)
        
        # Find all frame indices
        frame_indices = []
        for key in npz_data.keys():
            if key.startswith('frame_') and key.endswith('_frame_idx'):
                frame_idx = int(key.split('_')[1])
                if frame_idx not in frame_indices:
                    frame_indices.append(frame_idx)
        frame_indices.sort()
        
        deltas = []
        for i in frame_indices:
            prefix = f"frame_{i}"
            
            delta_mean = torch.from_numpy(npz_data[f"{prefix}_mean"])
            delta_scales = torch.from_numpy(npz_data[f"{prefix}_scales"])
            delta_quats = torch.from_numpy(npz_data[f"{prefix}_quats"])
            delta_colors = torch.from_numpy(npz_data[f"{prefix}_colors"])
            delta_opacities = torch.from_numpy(npz_data[f"{prefix}_opacities"])
            
            changed_indices = None
            if f"{prefix}_changed" in npz_data:
                changed_indices = torch.from_numpy(npz_data[f"{prefix}_changed"])
            
            base_match_indices = None
            if f"{prefix}_base_match" in npz_data:
                base_match_indices = torch.from_numpy(npz_data[f"{prefix}_base_match"])
            
            current_match_indices = None
            if f"{prefix}_current_match" in npz_data:
                current_match_indices = torch.from_numpy(npz_data[f"{prefix}_current_match"])
            
            is_new_gaussian = None
            if f"{prefix}_is_new" in npz_data:
                is_new_gaussian = torch.from_numpy(npz_data[f"{prefix}_is_new"]).bool()
            
            frame_idx = int(npz_data[f"{prefix}_frame_idx"][0])
            base_idx = int(npz_data[f"{prefix}_base_idx"][0])
            metadata_arr = npz_data[f"{prefix}_metadata"]
            
            delta_frame = DeltaFrame(
                frame_index=frame_idx,
                base_frame_index=base_idx,
                delta_mean=delta_mean,
                delta_scales=delta_scales,
                delta_quaternions=delta_quats,
                delta_colors=delta_colors,
                delta_opacities=delta_opacities,
                changed_indices=changed_indices,
                base_match_indices=base_match_indices,
                current_match_indices=current_match_indices,
                is_new_gaussian=is_new_gaussian,
            )
            
            metadata = SceneMetaData(
                focal_length_px=float(metadata_arr[0]),
                resolution_px=(int(metadata_arr[1]), int(metadata_arr[2])),
                color_space=base_metadata.color_space,
            )
            
            deltas.append((delta_frame, metadata))
        
        npz_data.close()
    
    # Try pickle format (legacy)
    elif delta_path.suffix in ['.pkl', '.gz', '.xz'] and 'pkl' in delta_path.stem:
        if delta_path.suffix == '.gz':
            with gzip.open(delta_path, 'rb') as f:
                deltas = pickle.load(f)
        elif delta_path.suffix == '.xz':
            with lzma.open(delta_path, 'rb') as f:
                deltas = pickle.load(f)
        else:
            with open(delta_path, 'rb') as f:
                deltas = pickle.load(f)
    else:
        # Fallback to JSON format (legacy)
        if delta_path.suffix == '.gz':
            with gzip.open(delta_path, 'rt') as f:
                deltas_data = json.load(f)
        elif delta_path.suffix == '.xz':
            with lzma.open(delta_path, 'rt') as f:
                deltas_data = json.load(f)
        else:
            with open(delta_path, 'r') as f:
                deltas_data = json.load(f)
        
        deltas = []
        for delta_dict in deltas_data:
            meta_dict = delta_dict.pop("metadata", {})
            delta_frame = DeltaFrame.from_dict(delta_dict)
            metadata = SceneMetaData(
                focal_length_px=meta_dict.get("focal_length_px", base_metadata.focal_length_px),
                resolution_px=tuple(meta_dict.get("resolution_px", base_metadata.resolution_px)),
                color_space=meta_dict.get("color_space", base_metadata.color_space),
            )
            deltas.append((delta_frame, metadata))
    
    for delta_frame, metadata in deltas:
        frame_idx = delta_frame.frame_index
        base_idx = delta_frame.base_frame_index
        
        if base_idx not in keyframes:
            LOGGER.warning(f"Missing base frame {base_idx} for frame {frame_idx}")
            continue
        
        base_gaussians, base_metadata = keyframes[base_idx]
        
        # Convert tensors if needed (from pickle they might be numpy arrays)
        if isinstance(delta_frame.delta_mean, np.ndarray):
            delta_mean = torch.from_numpy(delta_frame.delta_mean)
            delta_scales = torch.from_numpy(delta_frame.delta_scales)
            delta_quats = torch.from_numpy(delta_frame.delta_quaternions)
            delta_colors = torch.from_numpy(delta_frame.delta_colors)
            delta_opacities = torch.from_numpy(delta_frame.delta_opacities)
        else:
            delta_mean = delta_frame.delta_mean
            delta_scales = delta_frame.delta_scales
            delta_quats = delta_frame.delta_quaternions
            delta_colors = delta_frame.delta_colors
            delta_opacities = delta_frame.delta_opacities
        
        delta_gaussians = Gaussians3D(
            mean_vectors=delta_mean,
            singular_values=delta_scales,
            quaternions=delta_quats,
            colors=delta_colors,
            opacities=delta_opacities,
        )
        
        # Convert indices if needed
        changed_indices = delta_frame.changed_indices
        base_match_indices = delta_frame.base_match_indices
        current_match_indices = delta_frame.current_match_indices
        is_new_gaussian = delta_frame.is_new_gaussian
        
        if changed_indices is not None and isinstance(changed_indices, np.ndarray):
            changed_indices = torch.from_numpy(changed_indices)
        if base_match_indices is not None and isinstance(base_match_indices, np.ndarray):
            base_match_indices = torch.from_numpy(base_match_indices)
        if current_match_indices is not None and isinstance(current_match_indices, np.ndarray):
            current_match_indices = torch.from_numpy(current_match_indices)
        if is_new_gaussian is not None and isinstance(is_new_gaussian, np.ndarray):
            is_new_gaussian = torch.from_numpy(is_new_gaussian)
        
        reconstructed = apply_delta(
            base_gaussians,
            delta_gaussians,
            changed_indices,
            base_match_indices,
            current_match_indices,
            is_new_gaussian,
        )
        
        output_path = output_dir / f"{frame_idx:04d}.ply"
        save_ply(
            reconstructed,
            metadata.focal_length_px,
            metadata.resolution_px[::-1],
            output_path,
        )
        LOGGER.info(f"Decompressed delta frame {frame_idx}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compress Gaussian splat PLY files with quantization, pruning, "
                    "and delta encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic compression with quantization
  python compress_splats.py data/gaussians/ -o data/compressed/

  # Aggressive compression (quantization + pruning + gzip)
  python compress_splats.py data/gaussians/ -o compressed/ \\
      --quantize --prune --opacity-threshold 0.05 --compression gzip

  # Delta encoding for sequences (interframe compression)
  python compress_splats.py data/gaussians/ -o compressed/ --delta

  # Custom quantization levels
  python compress_splats.py data/gaussians/ -o compressed/ \\
      --position-bits 16 --color-bits 8 --scale-bits 16

  # Decompress a delta-encoded sequence
  python compress_splats.py --decompress compressed/ -o decompressed/
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Input PLY file or directory containing PLY files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path (file or directory). Default: input_compressed/",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # Compression method
    parser.add_argument(
        "--compression",
        type=str,
        choices=["none", "gzip", "lzma"],
        default="none",
        help="Compression method for output files (default: none)",
    )
    
    # Quantization options
    quant_group = parser.add_argument_group("Quantization options")
    quant_group.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Enable quantization (default: True)",
    )
    quant_group.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
        help="Disable quantization",
    )
    quant_group.add_argument(
        "--position-bits",
        type=int,
        default=16,
        choices=[8, 16, 32],
        help="Bits for position quantization (default: 16)",
    )
    quant_group.add_argument(
        "--color-bits",
        type=int,
        default=8,
        choices=[8, 16, 32],
        help="Bits for color quantization (default: 8)",
    )
    quant_group.add_argument(
        "--scale-bits",
        type=int,
        default=16,
        choices=[8, 16, 32],
        help="Bits for scale quantization (default: 16)",
    )
    quant_group.add_argument(
        "--rotation-bits",
        type=int,
        default=16,
        choices=[8, 16, 32],
        help="Bits for rotation quantization (default: 16)",
    )
    quant_group.add_argument(
        "--opacity-bits",
        type=int,
        default=8,
        choices=[8, 16, 32],
        help="Bits for opacity quantization (default: 8)",
    )
    
    # Pruning options
    prune_group = parser.add_argument_group("Pruning options")
    prune_group.add_argument(
        "--prune",
        action="store_true",
        help="Remove low-opacity Gaussians",
    )
    prune_group.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.01,
        help="Opacity threshold for pruning (default: 0.01)",
    )
    prune_group.add_argument(
        "--scale-threshold",
        type=float,
        default=None,
        help="Scale threshold for pruning (default: None)",
    )
    
    # Delta encoding options
    delta_group = parser.add_argument_group("Delta encoding options (interframe compression)")
    delta_group.add_argument(
        "--delta",
        action="store_true",
        help="Use delta encoding for sequences",
    )
    delta_group.add_argument(
        "--keyframe-interval",
        type=int,
        default=10,
        help="Insert keyframe every N frames (default: 10)",
    )
    delta_group.add_argument(
        "--sparse-threshold",
        type=float,
        default=1e-4,
        help="Threshold for sparse delta encoding (default: 1e-4)",
    )
    delta_group.add_argument(
        "--max-match-distance",
        type=float,
        default=0.1,
        help="Maximum 3D distance for spatial matching between frames (default: 0.1)",
    )
    delta_group.add_argument(
        "--threshold-percentile",
        type=float,
        default=50.0,
        help="Percentile for adaptive threshold (default: 50.0, i.e., median)",
    )
    delta_group.add_argument(
        "--no-adaptive-threshold",
        dest="adaptive_threshold",
        action="store_false",
        help="Disable adaptive thresholding (use fixed threshold)",
    )
    delta_group.add_argument(
        "--no-quantize-deltas",
        dest="quantize_deltas",
        action="store_false",
        help="Disable delta quantization",
    )
    
    # Decompress mode
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress a delta-encoded sequence instead of compressing",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging_utils.configure(logging.DEBUG if args.verbose else logging.INFO)
    
    # Handle decompression
    if args.decompress:
        output = args.output or args.input.parent / f"{args.input.name}_decompressed"
        decompress_sequence(args.input, output)
        return
    
    # Build quantization config
    quant_config = QuantizationConfig(
        position_bits=args.position_bits,
        color_bits=args.color_bits,
        scale_bits=args.scale_bits,
        rotation_bits=args.rotation_bits,
        opacity_bits=args.opacity_bits,
    )
    
    if args.input.is_file():
        # Single file compression
        output = args.output or args.input.parent / f"{args.input.stem}_compressed.ply"
        compress_single_file(
            args.input,
            output,
            quantize=args.quantize,
            quant_config=quant_config,
            prune=args.prune,
            opacity_threshold=args.opacity_threshold,
            scale_threshold=args.scale_threshold,
            compression=args.compression,
        )
    else:
        # Directory/sequence compression
        output = args.output or args.input.parent / f"{args.input.name}_compressed"
        compress_sequence(
            args.input,
            output,
            use_delta=args.delta,
            quantize=args.quantize,
            quant_config=quant_config,
            prune=args.prune,
            opacity_threshold=args.opacity_threshold,
            scale_threshold=args.scale_threshold,
            compression=args.compression,
            sparse_delta_threshold=args.sparse_threshold,
            keyframe_interval=args.keyframe_interval,
            max_match_distance=args.max_match_distance,
            quantize_deltas=getattr(args, 'quantize_deltas', True),
            adaptive_threshold=getattr(args, 'adaptive_threshold', True),
            threshold_percentile=getattr(args, 'threshold_percentile', 50.0),
        )


if __name__ == "__main__":
    main()

