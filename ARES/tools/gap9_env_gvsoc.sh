#!/bin/bash
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

# GAP9 environment setup for gvsoc simulation.
#
# Before sourcing this script, set the following environment variables
# (or edit the defaults below) to match your local GAP SDK installation:
#
#   GAP_SDK_HOME              Path to the GAP SDK root directory
#   GAP_RISCV_GCC_TOOLCHAIN   Path to the RISC-V GCC toolchain
#   GAP_SDK_CONDA_ENV          (optional) Conda environment for GAP SDK
#   CC / CXX                   (optional) Host C/C++ compiler

set -eo pipefail

# ---------------------------------------------------------------------------
# Conda (optional)
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if [ -n "${GAP_SDK_CONDA_ENV:-}" ] && [ -d "$GAP_SDK_CONDA_ENV" ]; then
    conda activate "$GAP_SDK_CONDA_ENV" >/dev/null
fi

# ---------------------------------------------------------------------------
# Python path guard
# ---------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_PY="$SCRIPT_DIR/vendor"
if [ -d "$VENDOR_PY" ]; then
    if ! python3 -c "import prettytable" >/dev/null 2>&1; then
        export PYTHONPATH="$VENDOR_PY:${PYTHONPATH:-}"
    fi
fi

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export GAP_RISCV_GCC_TOOLCHAIN="${GAP_RISCV_GCC_TOOLCHAIN:?Set GAP_RISCV_GCC_TOOLCHAIN to the RISC-V toolchain path}"
export GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-ocd-h.cfg

# ---------------------------------------------------------------------------
# GAP SDK
# ---------------------------------------------------------------------------
: "${GAP_SDK_HOME:?Set GAP_SDK_HOME to the GAP SDK root directory}"

pushd "$GAP_SDK_HOME" >/dev/null
# shellcheck disable=SC1091
source configs/gap9_evk_audio.sh
popd >/dev/null
