# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Shape matching logic for ARES optimization knowledge base.

This module provides functions to match actual layer shapes against
patterns stored in the knowledge base.
"""

from typing import Dict, Any, List, Optional, Tuple
import math

from .config_schema import OptimizationEntry, ShapePattern
from ..constants import SCALE_EPSILON, SHAPE_EPSILON


def value_matches(actual: Any, pattern: Any) -> bool:
    """
    Check if an actual value matches a pattern value.

    Pattern values can be:
      - None: matches anything
      - int/float: exact match required
      - [min, max]: value must be in range (inclusive)
      - bool: exact match for boolean flags
      - str: exact match for string values

    Args:
        actual: The actual value from the layer
        pattern: The pattern value to match against

    Returns:
        True if actual matches pattern, False otherwise
    """
    # None pattern matches anything
    if pattern is None:
        return True

    # List pattern means range [min, max]
    if isinstance(pattern, list) and len(pattern) == 2:
        min_val, max_val = pattern
        if actual is None:
            return False
        try:
            return min_val <= actual <= max_val
        except TypeError:
            return False

    # Boolean exact match
    if isinstance(pattern, bool):
        return actual == pattern

    # Numeric exact match (with small tolerance for floats)
    if isinstance(pattern, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(pattern, float) or isinstance(actual, float):
            return abs(actual - pattern) < SHAPE_EPSILON
        return actual == pattern

    # String exact match
    if isinstance(pattern, str):
        return actual == pattern

    # Default: exact match
    return actual == pattern


def shape_matches(actual_shape: Dict[str, Any],
                  pattern: Dict[str, Any]) -> bool:
    """
    Check if actual shape matches a pattern.

    All keys in the pattern must match the actual shape.
    Extra keys in actual_shape that aren't in pattern are ignored.

    Args:
        actual_shape: Dictionary of actual layer dimensions
        pattern: Dictionary of pattern values to match

    Returns:
        True if all pattern keys match, False otherwise

    Example:
        >>> actual = {"M": 32, "N": 256, "K": 64}
        >>> pattern = {"M": [1, 64], "N": [128, 512]}
        >>> shape_matches(actual, pattern)
        True
    """
    for key, pattern_value in pattern.items():
        actual_value = actual_shape.get(key)
        if not value_matches(actual_value, pattern_value):
            return False
    return True


def shape_distance(actual_shape: Dict[str, Any],
                   pattern: Dict[str, Any]) -> float:
    """
    Compute distance between actual shape and pattern.

    Lower distance = better match. Returns infinity if pattern doesn't match.

    The distance is computed as:
    - 0 for exact matches
    - Normalized distance within range for range patterns
    - Infinity for non-matches

    Args:
        actual_shape: Dictionary of actual layer dimensions
        pattern: Dictionary of pattern values to match

    Returns:
        Distance score (lower is better), or infinity if no match
    """
    if not shape_matches(actual_shape, pattern):
        return float('inf')

    total_distance = 0.0
    num_keys = 0

    for key, pattern_value in pattern.items():
        actual_value = actual_shape.get(key)

        if pattern_value is None:
            # Any match - no distance contribution
            continue

        if isinstance(pattern_value, list) and len(pattern_value) == 2:
            # Range pattern - compute normalized distance from center
            min_val, max_val = pattern_value
            if actual_value is not None and isinstance(actual_value, (int, float)):
                center = (min_val + max_val) / 2
                range_size = max_val - min_val
                if range_size > 0:
                    # Distance from center, normalized by range
                    dist = abs(actual_value - center) / range_size
                    total_distance += dist
                    num_keys += 1
        elif isinstance(pattern_value, (int, float)) and isinstance(actual_value, (int, float)):
            # Exact match - distance is 0
            num_keys += 1

    # Average distance (or 0 if no numeric comparisons)
    if num_keys > 0:
        return total_distance / num_keys
    return 0.0


def pattern_specificity(pattern: Dict[str, Any]) -> float:
    """
    Compute how specific a pattern is.

    More specific patterns (exact values, narrow ranges) get higher scores.
    Used for tie-breaking when multiple patterns match.

    Args:
        pattern: Dictionary of pattern values

    Returns:
        Specificity score (higher = more specific)
    """
    specificity = 0.0

    for key, value in pattern.items():
        if value is None:
            # Matches anything - low specificity
            specificity += 0.1
        elif isinstance(value, list) and len(value) == 2:
            # Range - medium specificity, inversely proportional to range size
            min_val, max_val = value
            range_size = max_val - min_val
            if range_size > 0:
                # Narrower range = higher specificity
                specificity += 1.0 / (1.0 + math.log1p(range_size))
            else:
                # Zero-width range is like exact match
                specificity += 2.0
        elif isinstance(value, bool):
            # Boolean flag - high specificity
            specificity += 2.0
        else:
            # Exact match - highest specificity
            specificity += 2.0

    return specificity


def find_matching_entries(actual_shape: Dict[str, Any],
                          entries: List[OptimizationEntry],
                          op_type: str) -> List[Tuple[OptimizationEntry, float]]:
    """
    Find all entries that match the given shape and op type.

    Args:
        actual_shape: Dictionary of actual layer dimensions
        entries: List of optimization entries to search
        op_type: Operation type to filter by

    Returns:
        List of (entry, score) tuples, sorted by score (higher = better)
    """
    matches = []

    for entry in entries:
        # Filter by op type
        if entry.op_type != op_type:
            continue

        # Check if shape matches
        pattern = entry.shape_pattern.pattern
        if shape_matches(actual_shape, pattern):
            # Compute score based on distance, specificity, and confidence
            distance = shape_distance(actual_shape, pattern)
            specificity = pattern_specificity(pattern)
            confidence = entry.confidence

            # Score: lower distance is better, higher specificity/confidence is better
            # We want high scores to be better, so negate distance
            if distance < float('inf'):
                score = (1.0 / (1.0 + distance)) * specificity * confidence
            else:
                score = 0.0

            matches.append((entry, score))

    # Sort by score descending (best first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def find_best_match(actual_shape: Dict[str, Any],
                    entries: List[OptimizationEntry],
                    op_type: str,
                    min_confidence: float = 0.0) -> Optional[OptimizationEntry]:
    """
    Find the best matching entry for a shape.

    Args:
        actual_shape: Dictionary of actual layer dimensions
        entries: List of optimization entries to search
        op_type: Operation type to filter by
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        Best matching OptimizationEntry, or None if no match found
    """
    matches = find_matching_entries(actual_shape, entries, op_type)

    # Filter by confidence
    matches = [(entry, score) for entry, score in matches
               if entry.confidence >= min_confidence]

    if matches:
        return matches[0][0]
    return None


def check_negative_results(op_type: str,
                           config: Dict[str, Any],
                           negative_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Check if a proposed config matches any negative results.

    Args:
        op_type: Operation type
        config: Proposed configuration to check
        negative_results: List of negative result records

    Returns:
        The matching negative result dict if found, None otherwise
    """
    for neg in negative_results:
        if neg.get("op_type") != op_type:
            continue

        attempted = neg.get("attempted_config", {})

        # Check if all keys in attempted_config match the proposed config
        all_match = True
        for key, value in attempted.items():
            if key == "description":
                continue  # Skip description field
            if config.get(key) != value:
                all_match = False
                break

        if all_match and attempted:
            return neg

    return None


def extract_shape_from_layer(layer_info: Dict[str, Any],
                             op_type: str) -> Dict[str, Any]:
    """
    Extract relevant shape dimensions from layer info based on op type.

    Args:
        layer_info: Layer information dictionary
        op_type: Operation type

    Returns:
        Dictionary of shape dimensions relevant to this op type
    """
    shape = {}

    if op_type == "linear_int8":
        # Linear: M (batch*seq), N (out_features), K (in_features)
        shape["M"] = layer_info.get("batch_tokens", layer_info.get("M", 1))
        shape["N"] = layer_info.get("out_features", layer_info.get("N"))
        shape["K"] = layer_info.get("in_features", layer_info.get("K"))

    elif op_type == "conv2d_int8":
        # Conv2D: kernel size, channels, stride, padding
        shape["kernel_h"] = layer_info.get("kernel_h", layer_info.get("kernel_size", [3, 3])[0] if isinstance(layer_info.get("kernel_size"), list) else layer_info.get("kernel_size", 3))
        shape["kernel_w"] = layer_info.get("kernel_w", layer_info.get("kernel_size", [3, 3])[1] if isinstance(layer_info.get("kernel_size"), list) else layer_info.get("kernel_size", 3))
        shape["in_channels"] = layer_info.get("in_channels")
        shape["out_channels"] = layer_info.get("out_channels")
        shape["stride_h"] = layer_info.get("stride_h", layer_info.get("stride", 1))
        shape["stride_w"] = layer_info.get("stride_w", layer_info.get("stride", 1))
        shape["padding"] = layer_info.get("padding", 0)

    elif op_type == "mhsa_int8":
        # MHSA: seq_len, embed_dim, num_heads, head_dim
        shape["seq_len"] = layer_info.get("seq_len")
        shape["embed_dim"] = layer_info.get("embed_dim")
        shape["num_heads"] = layer_info.get("num_heads")
        shape["head_dim"] = layer_info.get("head_dim")

    elif op_type == "cross_attention_int8":
        # Cross-attention: num_queries, kv_len, embed_dim
        shape["num_queries"] = layer_info.get("num_queries")
        shape["kv_len"] = layer_info.get("kv_len")
        shape["embed_dim"] = layer_info.get("embed_dim")
        shape["num_heads"] = layer_info.get("num_heads")

    elif op_type == "ssm_int8":
        # SSM: d_inner, dt_rank, seq_len
        shape["d_inner"] = layer_info.get("d_inner")
        shape["dt_rank"] = layer_info.get("dt_rank")
        shape["seq_len"] = layer_info.get("seq_len")

    elif op_type in ("layernorm_int8", "gelu_int8"):
        # Normalization/activation: tokens, dim
        shape["tokens"] = layer_info.get("tokens", layer_info.get("seq_len"))
        shape["dim"] = layer_info.get("dim", layer_info.get("embed_dim"))

    elif op_type == "add_int8":
        # Element-wise: size
        shape["size"] = layer_info.get("size", layer_info.get("num_elements"))

    elif op_type == "identity_requant":
        # Identity requant: scale comparison
        scale_in = layer_info.get("scale_in")
        scale_out = layer_info.get("scale_out")
        if scale_in is not None and scale_out is not None:
            shape["scale_in_equals_scale_out"] = abs(scale_in - scale_out) < SCALE_EPSILON

    elif op_type == "groupnorm_int8":
        # GroupNorm: num_groups, num_channels, spatial_size
        shape["num_groups"] = layer_info.get("num_groups", layer_info.get("groups"))
        shape["num_channels"] = layer_info.get("num_channels", layer_info.get("channels"))
        shape["spatial_size"] = layer_info.get("spatial_size", layer_info.get("num_elements"))

    elif op_type == "rfft_int8":
        # RFFT: fft_size, num_features
        shape["fft_size"] = layer_info.get("fft_size", layer_info.get("n_fft"))
        shape["num_features"] = layer_info.get("num_features", layer_info.get("features"))

    elif op_type == "embedding":
        # Embedding: vocab_size, embed_dim, seq_len
        shape["vocab_size"] = layer_info.get("vocab_size", layer_info.get("num_embeddings"))
        shape["embed_dim"] = layer_info.get("embed_dim", layer_info.get("embedding_dim"))
        shape["seq_len"] = layer_info.get("seq_len")

    elif op_type == "rope_int8":
        # RoPE: seq_len, head_dim, num_heads
        shape["seq_len"] = layer_info.get("seq_len")
        shape["head_dim"] = layer_info.get("head_dim")
        shape["num_heads"] = layer_info.get("num_heads")

    # Remove None values
    shape = {k: v for k, v in shape.items() if v is not None}

    return shape
