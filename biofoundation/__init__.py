#*----------------------------------------------------------------------------*
#* Copyright (C) 2026 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  BioFoundation Contributors                                       *
#*----------------------------------------------------------------------------*

"""Shared infrastructure for the BioFoundation model zoo.

This package holds the contracts that model families, tasks, and datasets agree on:
the batch layout, the encoder and prediction-head protocols, checkpoint entry points,
environment validation, and the model registry. Everything here is imported by code
that must run without PyTorch installed, so this package has no heavyweight runtime
dependencies.

Changes follow an additive rule. New fields, functions, and classes may be added;
existing names do not change signature or behaviour, and defaults are chosen so that
an existing caller observes no difference. A change that cannot be made additively is
a major version bump and needs a deprecation period first.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]

