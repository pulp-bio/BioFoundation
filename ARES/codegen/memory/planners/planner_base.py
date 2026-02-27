# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Planner policy interface for L2 arena allocation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .planner_result import PlannerResult


class PlannerPolicyError(ValueError):
    """Raised when planner policy configuration is invalid."""


class PlannerBase(ABC):
    """Abstract base class for planner policies."""

    policy_name = "planner_base"
    is_experimental = False

    @abstractmethod
    def plan(
        self,
        specs: List[Dict[str, Any]],
        activation_buffers: List[Dict[str, Any]],
        shared_pool: List[Dict[str, Any]],
    ) -> PlannerResult:
        """Return placement/lifetime result for an arena allocation policy."""
