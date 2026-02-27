# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Base interface for codegen pipeline passes."""

from abc import ABC, abstractmethod

from .context import PipelineContext


class CodegenPass(ABC):
    """Base class for all codegen passes."""

    name = "unnamed_pass"

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        """Execute pass logic and mutate the shared pipeline context."""
