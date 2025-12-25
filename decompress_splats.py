#!/usr/bin/env python3
"""Decompress Gaussian splat PLY files from compressed formats.

This script can decompress:
1. Delta-encoded sequences (with keyframes and deltas)
2. Individual compressed PLY files (gzip/lzma)
3. Quantized/pruned PLY files

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
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sharp.utils.gaussians import load_ply, save_ply, Gaussians3D, SceneMetaData
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)


# Reuse DeltaFrame from compress_splats if available, otherwise define it
try:
    from compress_splats import DeltaFrame
except ImportError:
    from dataclasses import dataclass
    from typing import Optional as Opt
    
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
        changed_indices: Opt[torch.Tensor] = None
        base_match_indices: Opt[torch.Tensor] = None
        current_match_indices: Opt[torch.Tensor] = None
        is_new_gaussian: Opt[torch.Tensor] = None


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


def decompress_delta_sequence(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Decompress a delta-encoded sequence back to individual PLY files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Decompressing delta-encoded sequence from {input_dir}")
    
    # Load keyframes
    keyframe_files = sorted(input_dir.glob("keyframe_*.ply*"))
    if not keyframe_files:
        LOGGER.error(f"No keyframes found in {input_dir}")
        return
    
    keyframes = {}
    base_metadata = None
    
    for kf_path in keyframe_files:
        # Extract frame index from filename
        stem = kf_path.stem.replace('.ply', '')
        frame_idx = int(stem.split('_')[1])
        gaussians, metadata = load_compressed_ply(kf_path)
        keyframes[frame_idx] = (gaussians, metadata)
        
        if base_metadata is None:
            base_metadata = metadata
        
        # Save decompressed keyframe
        output_path = output_dir / f"{frame_idx:04d}.ply"
        save_ply(gaussians, metadata.focal_length_px, metadata.resolution_px[::-1], output_path)
        LOGGER.info(f"Decompressed keyframe {frame_idx} -> {output_path}")
    
    # Load and apply deltas (try numpy format first, then pickle, then JSON)
    delta_files = list(input_dir.glob("deltas.npz")) + list(input_dir.glob("deltas.pkl*")) + list(input_dir.glob("deltas.json*"))
    if not delta_files:
        LOGGER.info("No delta files found, only keyframes decompressed")
        return
    
    delta_path = delta_files[0]
    LOGGER.info(f"Loading deltas from {delta_path}")
    
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
        
        LOGGER.info(f"Found {len(frame_indices)} delta frames")
        
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
    
    # Apply deltas to reconstruct frames
    LOGGER.info(f"Reconstructing {len(deltas)} frames from deltas...")
    
    for delta_frame, metadata in deltas:
        frame_idx = delta_frame.frame_index
        base_idx = delta_frame.base_frame_index
        
        if base_idx not in keyframes:
            LOGGER.warning(f"Missing base frame {base_idx} for frame {frame_idx}, skipping")
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
        LOGGER.info(f"Decompressed delta frame {frame_idx} -> {output_path}")
    
    LOGGER.info(f"Decompression complete! Output saved to {output_dir}")


def decompress_single_file(
    input_path: Path,
    output_path: Path,
) -> None:
    """Decompress a single compressed PLY file."""
    LOGGER.info(f"Decompressing {input_path} -> {output_path}")
    
    gaussians, metadata = load_compressed_ply(input_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(gaussians, metadata.focal_length_px, metadata.resolution_px[::-1], output_path)
    
    LOGGER.info(f"Decompressed {input_path} -> {output_path}")


def decompress_directory(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Decompress all compressed PLY files in a directory."""
    LOGGER.info(f"Decompressing all files from {input_dir} -> {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all compressed PLY files
    compressed_files = (
        list(input_dir.glob("*.ply.gz")) +
        list(input_dir.glob("*.ply.xz")) +
        list(input_dir.glob("*.ply"))
    )
    
    if not compressed_files:
        LOGGER.warning(f"No PLY files found in {input_dir}")
        return
    
    for compressed_file in sorted(compressed_files):
        # Remove compression suffix for output filename
        output_name = compressed_file.stem.replace('.ply', '') + '.ply'
        output_path = output_dir / output_name
        
        decompress_single_file(compressed_file, output_path)
    
    LOGGER.info(f"Decompressed {len(compressed_files)} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Decompress Gaussian splat PLY files from compressed formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decompress a delta-encoded sequence
  python decompress_splats.py compressed/ -o decompressed/

  # Decompress a single compressed file
  python decompress_splats.py compressed/file.ply.gz -o decompressed/file.ply

  # Decompress all files in a directory
  python decompress_splats.py compressed/ -o decompressed/ --all
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Input compressed file or directory containing compressed files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output path (file or directory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Decompress all files in directory (not just delta sequences)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging_utils.configure(logging.DEBUG if args.verbose else logging.INFO)
    
    if args.input.is_file():
        # Single file decompression
        if not args.output.suffix:
            args.output = args.output.with_suffix('.ply')
        decompress_single_file(args.input, args.output)
    else:
        # Directory decompression
        if args.all:
            # Decompress all files in directory
            decompress_directory(args.input, args.output)
        else:
            # Try delta sequence decompression first
            if any(args.input.glob("deltas.*")) or any(args.input.glob("keyframe_*")):
                decompress_delta_sequence(args.input, args.output)
            else:
                # Fallback to directory decompression
                LOGGER.info("No delta sequence found, decompressing all files...")
                decompress_directory(args.input, args.output)


if __name__ == "__main__":
    main()

