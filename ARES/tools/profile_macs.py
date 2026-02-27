#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LayerInfo:
    name: str
    op_type: str
    macs: int | None


@dataclass(frozen=True)
class LayerPerf:
    name: str
    total_cycles: int
    compute_cycles: int
    dma_load_cycles: int
    dma_store_cycles: int
    idle_cycles: int
    overlap_percent: float


_RE_LAYER_TYPE = re.compile(r"\.type\s*=\s*(OP_[A-Z0-9_]+)")
_RE_LAYER_NAME = re.compile(r'\.name\s*=\s*"([^"]+)"')
_RE_NETWORK_LAYERS_DECL = re.compile(
    r"\b(?:static\s+)?(?:const\s+)?LayerSpec\s+network_layers\s*\[\s*\]\s*=\s*\{",
    flags=re.MULTILINE,
)


def _extract_network_layers_array(text: str) -> str:
    match = _RE_NETWORK_LAYERS_DECL.search(text)
    if not match:
        raise ValueError("Could not find `LayerSpec network_layers[] = { ... }` initializer in generated C")

    brace_open_idx = match.end() - 1

    depth = 0
    for idx in range(brace_open_idx, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_open_idx + 1 : idx]

    raise ValueError("Could not find end of network_layers array initializer")


def _split_top_level_blocks(array_body: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    block_start = None

    for idx, ch in enumerate(array_body):
        if ch == "{":
            if depth == 0:
                block_start = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and block_start is not None:
                blocks.append(array_body[block_start : idx + 1])
                block_start = None

    return blocks


def _search_int(block: str, field: str) -> int:
    match = re.search(rf"\.{re.escape(field)}\s*=\s*(-?\d+)", block)
    if not match:
        raise ValueError(f"Missing field `{field}`")
    return int(match.group(1))


def _parse_layer_macs_from_block(block: str) -> LayerInfo | None:
    type_match = _RE_LAYER_TYPE.search(block)
    name_match = _RE_LAYER_NAME.search(block)
    if not type_match or not name_match:
        return None

    op_type = type_match.group(1)
    name = name_match.group(1)

    macs: int | None = None
    try:
        if op_type == "OP_CONV2D":
            out_h = _search_int(block, "out_h")
            out_w = _search_int(block, "out_w")
            out_ch = _search_int(block, "out_ch")
            kernel_h = _search_int(block, "kernel_h")
            kernel_w = _search_int(block, "kernel_w")
            in_ch = _search_int(block, "in_ch")
            macs = out_h * out_w * out_ch * kernel_h * kernel_w * in_ch
        elif op_type == "OP_LINEAR_INT8":
            in_features = _search_int(block, "in_features")
            out_features = _search_int(block, "out_features")
            batch_tokens = _search_int(block, "batch_tokens")
            macs = batch_tokens * in_features * out_features
        elif op_type == "OP_LINEAR_FP32":
            in_features = _search_int(block, "in_features")
            out_features = _search_int(block, "out_features")
            macs = in_features * out_features
        elif op_type == "OP_MHSA":
            seq_len = _search_int(block, "seq_len")
            num_heads = _search_int(block, "num_heads")
            head_dim = _search_int(block, "head_dim")
            embed_dim = _search_int(block, "embed_dim")

            qkv_proj = 3 * seq_len * embed_dim * embed_dim
            qk = num_heads * seq_len * seq_len * head_dim
            av = qk
            out_proj = seq_len * embed_dim * embed_dim
            macs = qkv_proj + qk + av + out_proj
        elif op_type == "OP_CROSS_ATTENTION":
            batch = _search_int(block, "batch")
            kv_len = _search_int(block, "kv_len")
            num_queries = _search_int(block, "num_queries")
            num_heads = _search_int(block, "num_heads")
            head_dim = _search_int(block, "head_dim")
            embed_dim = _search_int(block, "embed_dim")

            # Projections:
            # - Q: [B, Q, D] x [D, D]
            # - K/V: [B, N, D] x [D, D]
            # - OUT: [B, Q, D] x [D, D]
            q_proj = batch * num_queries * embed_dim * embed_dim
            k_proj = batch * kv_len * embed_dim * embed_dim
            v_proj = k_proj
            out_proj = batch * num_queries * embed_dim * embed_dim

            # Attention:
            # - QK: [B, H, Q, Dh] x [B, H, N, Dh]^T
            # - AV: [B, H, Q, N] x [B, H, N, Dh]
            qk = batch * num_heads * num_queries * kv_len * head_dim
            av = qk

            macs = q_proj + k_proj + v_proj + qk + av + out_proj
    except ValueError:
        macs = None

    return LayerInfo(name=name, op_type=op_type, macs=macs)


def parse_network_layers(network_c: Path) -> list[LayerInfo]:
    text = network_c.read_text(encoding="utf-8", errors="replace")
    array_body = _extract_network_layers_array(text)
    blocks = _split_top_level_blocks(array_body)

    layers: list[LayerInfo] = []
    for block in blocks:
        parsed = _parse_layer_macs_from_block(block)
        if parsed is not None:
            layers.append(parsed)
    return layers


_RE_PERF_LINE = re.compile(
    r"^\s*PERF\s+(?P<name>[^:]+?)\s*:\s*"
    r"total=\s*(?P<total>\d+)\s+"
    r"compute=\s*(?P<compute>\d+)\s+"
    r"dma_load=\s*(?P<dma_load>\d+)\s+"
    r"dma_store=\s*(?P<dma_store>\d+)\s+"
    r"idle=\s*(?P<idle>\d+)\s+"
    r"overlap=\s*(?P<overlap>[0-9.]+)%\s*$"
)


def parse_perf_log(perf_log: Path) -> dict[str, LayerPerf]:
    per_layer: dict[str, LayerPerf] = {}
    for line in perf_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RE_PERF_LINE.match(line)
        if not match:
            continue

        name = match.group("name").strip()
        per_layer[name] = LayerPerf(
            name=name,
            total_cycles=int(match.group("total")),
            compute_cycles=int(match.group("compute")),
            dma_load_cycles=int(match.group("dma_load")),
            dma_store_cycles=int(match.group("dma_store")),
            idle_cycles=int(match.group("idle")),
            overlap_percent=float(match.group("overlap")),
        )
    return per_layer


def _format_int(n: int) -> str:
    return f"{n:,}"


def _format_float(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(r: Iterable[str]) -> str:
        return "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(r))

    print(fmt_row(headers))
    print(fmt_row("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def _print_markdown_table(rows: list[list[str]], headers: list[str]) -> None:
    def md_cell(value: str, *, code: bool = False) -> str:
        if value == "-":
            return "-"
        if code:
            return f"`{value}`"
        return value

    md_headers = [
        md_cell(headers[0]),
        md_cell(headers[1]),
        md_cell(headers[2]),
        md_cell(headers[3]),
        md_cell(headers[4]),
        md_cell(headers[5]),
        md_cell(headers[6]),
    ]
    print("| " + " | ".join(md_headers) + " |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        md_row = [
            md_cell(row[0], code=True),
            md_cell(row[1], code=True),
            md_cell(row[2]),
            md_cell(row[3]),
            md_cell(row[4]),
            md_cell(row[5]),
            md_cell(row[6]),
        ]
        print("| " + " | ".join(md_row) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-layer MACs and MACs/cycle from generated network layer specs and perf logs."
    )
    parser.add_argument(
        "--network-c",
        type=Path,
        required=True,
        help="Path to generated C file containing `LayerSpec network_layers[] = { ... }`",
    )
    parser.add_argument("--perf-log", type=Path, help="Path to gvsoc perf log (gvsoc_run_perf.log)")
    parser.add_argument("--format", choices=["text", "markdown"], default="text", help="Output format (default: text)")
    args = parser.parse_args()

    layers = parse_network_layers(args.network_c)
    total_macs = sum(layer.macs or 0 for layer in layers)

    perf_by_name = parse_perf_log(args.perf_log) if args.perf_log else {}

    rows: list[list[str]] = []
    for layer in layers:
        perf = perf_by_name.get(layer.name)
        total_cycles = perf.total_cycles if perf else 0
        compute_cycles = perf.compute_cycles if perf else 0

        macs_per_total = (layer.macs / total_cycles) if layer.macs and total_cycles > 0 else None
        macs_per_compute = (layer.macs / compute_cycles) if layer.macs and compute_cycles > 0 else None

        rows.append(
            [
                layer.name,
                layer.op_type,
                _format_int(layer.macs) if layer.macs else "-",
                _format_int(total_cycles) if perf else "-",
                _format_int(compute_cycles) if perf else "-",
                _format_float(macs_per_total),
                _format_float(macs_per_compute),
            ]
        )

    headers = ["layer", "op", "MACs", "total cyc", "compute cyc", "MACs/total", "MACs/compute"]
    if args.format == "markdown":
        _print_markdown_table(rows, headers)
    else:
        _print_table(rows, headers)

    print()
    print(f"Total MACs: {_format_int(total_macs)}")
    if perf_by_name:
        total_perf_cycles = sum(p.total_cycles for p in perf_by_name.values())
        if total_perf_cycles > 0:
            print(f"Total perf cycles (sum of PERF layers): {_format_int(total_perf_cycles)}")
            print(f"Overall MACs/total-cycle (perf-summed): {total_macs / total_perf_cycles:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
