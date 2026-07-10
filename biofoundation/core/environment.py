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

"""Environment validation shared by command-line entry points."""

import os
from collections.abc import Mapping, Sequence


def missing_environment_variables(
    names: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return required variables that are unset or still placeholders."""

    values = os.environ if environ is None else environ
    return tuple(name for name in names if not values.get(name) or values.get(name) == "#CHANGEME")


def require_environment(
    names: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> None:
    """Raise a single actionable error for missing runtime environment values."""

    missing = missing_environment_variables(names, environ)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Required environment variables are not set: {joined}")

