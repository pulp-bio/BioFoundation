# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Arena planner policies (default + experimental scaffold)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .planner_base import PlannerBase
from .planner_result import PlannerResult


class ArenaFirstFitPlanner(PlannerBase):
    """First-fit arena planner used as the default memory-planning policy."""

    policy_name = "arena_first_fit"
    is_experimental = False

    def plan(
        self,
        specs: List[Dict[str, Any]],
        activation_buffers: List[Dict[str, Any]],
        shared_pool: List[Dict[str, Any]],
    ) -> PlannerResult:
        buffers = self._build_buffer_map(activation_buffers, shared_pool)
        lifetimes = self._compute_lifetimes(specs, buffers)
        offsets, total_size = self._allocate_first_fit(lifetimes)

        return PlannerResult(
            policy=self.policy_name,
            lifetimes=lifetimes,
            offsets=offsets,
            total_size=total_size,
            unresolved_conflicts=[],
        )

    def _build_buffer_map(
        self,
        activation_buffers: List[Dict[str, Any]],
        shared_pool: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        buffers = {b["name"]: b for b in activation_buffers}
        for pool in shared_pool:
            buffers[pool["name"]] = pool
        return buffers

    def _buffer_size_bytes(self, buf: Dict[str, Any]) -> int:
        ctype = buf.get("ctype", "int8_t")
        if ctype in ("int32_t", "uint32_t", "float", "fp32"):
            elem_size = 4
        elif ctype in ("int16_t", "uint16_t"):
            elem_size = 2
        else:
            elem_size = 1
        return int(buf.get("numel", 0)) * elem_size

    def _find_buf_by_cname(self, c_name: Any, buffers: Dict[str, Dict[str, Any]]) -> str | None:
        if not c_name or not isinstance(c_name, str):
            return None

        if c_name.startswith("blocks_"):
            parts = c_name.split("_", 2)
            if len(parts) >= 3:
                role_with_suffix = parts[2]
                role = role_with_suffix[:-4] if role_with_suffix.endswith("_out") else role_with_suffix
                if role in ("norm1", "norm2"):
                    role = "norm"
                pool_cname = f"block_{role}_out_pool"
                for name, buf in buffers.items():
                    if buf.get("c_name") == pool_cname:
                        return name

        for name, buf in buffers.items():
            if buf.get("c_name") == c_name:
                return name
        return None

    def _iter_input_cnames(self, spec: Dict[str, Any]) -> List[str]:
        inputs: List[str] = []
        if "input_buffer" in spec:
            inputs.append(spec["input_buffer"])
        if "input1_buffer" in spec:
            inputs.append(spec["input1_buffer"])
        if "input2_buffer" in spec:
            inputs.append(spec["input2_buffer"])
        if "input_buffers" in spec:
            inputs.extend(spec.get("input_buffers") or [])
        if spec.get("op") == "mhsa":
            inputs.extend([spec.get("q_buffer"), spec.get("k_buffer"), spec.get("v_buffer")])
        return inputs

    def _compute_lifetimes(
        self,
        specs: List[Dict[str, Any]],
        buffers: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, int]]:
        lifetimes: Dict[str, Dict[str, int]] = {}
        for name, buf in buffers.items():
            if buf.get("use_l3_fallback", False):
                continue
            lifetimes[name] = {"start": -1, "end": -1, "size": self._buffer_size_bytes(buf)}

        pool_per_block_lifetimes: Dict[str, Dict[int, List[int]]] = {}

        for i, spec in enumerate(specs):
            block_id = spec.get("block_id")

            for c_name in self._iter_input_cnames(spec):
                buf_name = self._find_buf_by_cname(c_name, buffers)
                if not buf_name or buf_name not in lifetimes:
                    continue
                is_pool = buffers.get(buf_name, {}).get("is_pool")
                if is_pool and block_id is not None:
                    pool_per_block_lifetimes.setdefault(buf_name, {}).setdefault(block_id, [i, i])[1] = i
                else:
                    lifetimes[buf_name]["end"] = i

            output = spec.get("output_buffer")
            if output:
                buf_name = self._find_buf_by_cname(output, buffers)
                if buf_name and buf_name in lifetimes:
                    is_pool = buffers.get(buf_name, {}).get("is_pool")
                    if is_pool and block_id is not None:
                        block_range = pool_per_block_lifetimes.setdefault(buf_name, {}).setdefault(block_id, [i, i])
                        block_range[1] = i
                    else:
                        if lifetimes[buf_name]["start"] == -1:
                            lifetimes[buf_name]["start"] = i
                        lifetimes[buf_name]["end"] = max(lifetimes[buf_name]["end"], i)

            if spec.get("op") == "mhsa":
                for c_name in (spec.get("q_buffer"), spec.get("k_buffer"), spec.get("v_buffer")):
                    buf_name = self._find_buf_by_cname(c_name, buffers)
                    if not buf_name or buf_name not in lifetimes:
                        continue
                    is_pool = buffers.get(buf_name, {}).get("is_pool")
                    if is_pool and block_id is not None:
                        pool_per_block_lifetimes.setdefault(buf_name, {}).setdefault(block_id, [i, i])[1] = i
                    else:
                        if lifetimes[buf_name]["start"] == -1:
                            lifetimes[buf_name]["start"] = i
                        lifetimes[buf_name]["end"] = max(lifetimes[buf_name]["end"], i)

        for pool_name, blocks in pool_per_block_lifetimes.items():
            if pool_name not in lifetimes:
                continue
            max_span = -1
            best_start = -1
            best_end = -1
            for start, end in blocks.values():
                span = end - start
                if span > max_span:
                    max_span = span
                    best_start = start
                    best_end = end
            if best_start >= 0:
                lifetimes[pool_name]["start"] = best_start
                lifetimes[pool_name]["end"] = best_end

        if "input_quant" in lifetimes:
            lifetimes["input_quant"]["start"] = 0

        final_step = len(specs)
        for life in lifetimes.values():
            if life["end"] < life["start"]:
                life["end"] = final_step

        for life in lifetimes.values():
            if life["start"] == -1:
                life["start"] = 0
                life["end"] = final_step

        return lifetimes

    def _sort_for_allocation(self, lifetimes: Dict[str, Dict[str, int]]) -> List[Tuple[str, Dict[str, int]]]:
        return sorted(lifetimes.items(), key=lambda item: (item[1]["start"], -item[1]["size"]))

    def _allocate_first_fit(
        self,
        lifetimes: Dict[str, Dict[str, int]],
    ) -> Tuple[Dict[str, int], int]:
        offsets: Dict[str, int] = {}
        total_size = 0
        allocated_blocks: List[Tuple[int, int, int, int]] = []

        for name, life in self._sort_for_allocation(lifetimes):
            size = life["size"]
            start = life["start"]
            end = life["end"]

            busy_ranges: List[Tuple[int, int]] = []
            for off, sz, s, e in allocated_blocks:
                if not (end < s or start > e):
                    busy_ranges.append((off, off + sz))
            busy_ranges.sort()

            candidate_addr = 0
            for b_start, b_end in busy_ranges:
                if candidate_addr + size <= b_start:
                    break
                candidate_addr = max(candidate_addr, b_end)

            offsets[name] = candidate_addr
            allocated_blocks.append((candidate_addr, size, start, end))
            total_size = max(total_size, candidate_addr + size)

        if total_size % 4 != 0:
            total_size += 4 - (total_size % 4)

        return offsets, total_size


class SizePriorityPlanner(ArenaFirstFitPlanner):
    """Experimental scaffold policy (disabled by default)."""

    policy_name = "size_priority"
    is_experimental = True

    def _sort_for_allocation(self, lifetimes: Dict[str, Dict[str, int]]) -> List[Tuple[str, Dict[str, int]]]:
        return sorted(lifetimes.items(), key=lambda item: (-item[1]["size"], item[1]["start"]))
