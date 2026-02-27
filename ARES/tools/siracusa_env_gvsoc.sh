#!/usr/bin/env bash
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

# Siracusa environment setup for gvsoc simulation.
#
# Before sourcing this script, set the following environment variables:
#
#   GVSOC_INSTALL_DIR          Path to gvsoc install directory
#   PULP_RISCV_GCC_TOOLCHAIN   Path to the PULP RISC-V GCC toolchain
#   PULP_SDK_HOME              Path to the pulp-sdk root directory

set -eo pipefail

# ---------------------------------------------------------------------------
# Conda (optional)
# ---------------------------------------------------------------------------
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate gvsoc
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate gvsoc
fi

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
: "${GVSOC_INSTALL_DIR:?Set GVSOC_INSTALL_DIR to the gvsoc install directory}"
: "${PULP_RISCV_GCC_TOOLCHAIN:?Set PULP_RISCV_GCC_TOOLCHAIN to the PULP RISC-V GCC toolchain path}"
: "${PULP_SDK_HOME:?Set PULP_SDK_HOME to the pulp-sdk root directory}"

export PATH="${GVSOC_INSTALL_DIR}/bin:${PATH}"
export PATH="${PULP_RISCV_GCC_TOOLCHAIN}/bin:${PATH}"
export PULP_RISCV_GCC_TOOLCHAIN

# ---------------------------------------------------------------------------
# Siracusa target config
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${PULP_SDK_HOME}/configs/siracusa.sh"
