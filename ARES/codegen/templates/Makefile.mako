# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

CORE ?= 8
ENABLE_PERF ?= 0
% if target_name == "siracusa":
USE_MCHAN_LOC_STRIDE ?= 0
% else:
USE_MCHAN_LOC_STRIDE ?= 1
% endif
% if ne16_eligible_layers:
USE_NE16 ?= 1
% else:
USE_NE16 ?= 0
% endif
# NE16 hardware requantization (OUTQUANT): produces int8 directly (not golden-exact vs SW nearest-even)
# Enable with: make USE_NE16=1 USE_NE16_HW_REQUANT=1
USE_NE16_HW_REQUANT ?= 0
# Warning control: WARNINGS=1 (default) enables -Wall, WARNINGS=0 uses minimal suppressions
# Both modes suppress unused variable/function warnings; WARNINGS=1 adds -Wall for extra checks
WARNINGS ?= 1
EXTRA_CFLAGS ?=
EXTRA_LDFLAGS ?=
PI_CL_SLAVE_STACK_SIZE ?= 0x400
% if board_mode:
# Board mode: minimal output enabled by default for clean timing
MINIMAL_OUTPUT ?= 1
DISABLE_INTERMEDIATE_GOLDEN ?= 1
% else:
# Default generated projects to minimal runtime logs.
# Regression tooling overrides this with `MINIMAL_OUTPUT=0` when full
# validation output is required.
MINIMAL_OUTPUT ?= 1
% endif

APP = simplecnn_int8
# Locate repo root (directory containing `codegen/runtime/`) so generated projects can
# compile shared runtime sources without copying them into each `generated/` folder.
ROOT_CANDIDATES := \
  $(abspath $(CURDIR)) \
  $(abspath $(CURDIR)/..) \
  $(abspath $(CURDIR)/../..) \
  $(abspath $(CURDIR)/../../..) \
  $(abspath $(CURDIR)/../../../..) \
  $(abspath $(CURDIR)/../../../../..) \
  $(abspath $(CURDIR)/../../../../../..) \
  $(abspath $(CURDIR)/../../../../../../..) \
  $(abspath $(CURDIR)/../../../../../../../..) \
  $(abspath $(CURDIR)/../../../../../../../../..)
PROJECT_ROOT := $(firstword $(foreach d,$(ROOT_CANDIDATES),$(if $(wildcard $(d)/codegen/runtime),$(d),)))
ifeq ($(strip $(PROJECT_ROOT)),)
$(error Could not locate repo root (missing codegen/runtime). Set PROJECT_ROOT manually or run from inside the repo.)
endif
RUNTIME_DIR := $(PROJECT_ROOT)/codegen/runtime

APP_SRCS := \
  $(wildcard src/*.c) \
  $(wildcard src/net/*.c) \
  $(wildcard src/ops/*.c) \
  $(wildcard $(RUNTIME_DIR)/src/*.c) \
  $(wildcard $(RUNTIME_DIR)/src/core/*.c) \
  $(wildcard $(RUNTIME_DIR)/src/ops/*.c)

# NE16 accelerator sources
ifeq ($(USE_NE16),1)
APP_SRCS += $(wildcard $(RUNTIME_DIR)/src/ne16/*.c)
endif
# Prefer runtime headers over any stale generated duplicates.
APP_CFLAGS += -DNUM_CORES=$(CORE) -I$(RUNTIME_DIR)/inc -Iinc -O3 -fno-indirect-inlining -flto
# Warning flags: controlled by WARNINGS variable (default=1 for safety)
ifeq ($(WARNINGS),1)
APP_CFLAGS += -Wall -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable
else
APP_CFLAGS += -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable
endif
APP_CFLAGS += -DPI_CL_SLAVE_STACK_SIZE=$(PI_CL_SLAVE_STACK_SIZE)
APP_LDFLAGS += -lm -Wl,--print-memory-usage -flto
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM

APP_CFLAGS += -DGAP_SDK=1
APP_CFLAGS += -DFLASH_TYPE=$(FLASH_TYPE) -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS
APP_CFLAGS += $(EXTRA_CFLAGS)
APP_LDFLAGS += $(EXTRA_LDFLAGS)

# Performance counters - disabled by default
# Enable with: make ENABLE_PERF=1
ifeq ($(ENABLE_PERF),1)
APP_CFLAGS += -DENABLE_PERF_COUNTERS
endif

ifeq ($(USE_MCHAN_LOC_STRIDE),1)
APP_CFLAGS += -DUSE_MCHAN_LOC_STRIDE
endif

# Disable intermediate golden validation for large models to save L2 memory
# Enable with: make DISABLE_INTERMEDIATE_GOLDEN=1
# This only validates final output, not per-layer outputs
ifeq ($(DISABLE_INTERMEDIATE_GOLDEN),1)
APP_CFLAGS += -DDISABLE_INTERMEDIATE_GOLDEN
endif

# Minimal output mode for hardware timing - skips ALL golden checks and reduces prints
# Enable with: make MINIMAL_OUTPUT=1
# Use this when running on real hardware to get clean cycle counts
ifeq ($(MINIMAL_OUTPUT),1)
APP_CFLAGS += -DMINIMAL_OUTPUT
endif

# NE16 accelerator support
# Enable with: make USE_NE16=1
ifeq ($(USE_NE16),1)
APP_CFLAGS += -DARES_USE_NE16
ifeq ($(USE_NE16_HW_REQUANT),1)
APP_CFLAGS += -DARES_NE16_HW_REQUANT
endif
ifeq ($(USE_NE16_DEPTHWISE),1)
APP_CFLAGS += -DARES_NE16_DEPTHWISE
endif
endif

% if use_llama:
USE_LLAMA ?= 1
% else:
USE_LLAMA ?= 0
% endif
# Llama/LLM support (conditional to avoid code bloat for non-Llama models)
# Enable with: make USE_LLAMA=1
ifeq ($(USE_LLAMA),1)
APP_CFLAGS += -DARES_LLAMA_SUPPORT
endif

READFS_FILES := $(wildcard bin/*)
APP_CFLAGS += -DFS_READ_FS

include $(RULES_DIR)/pmsis_rules.mk
