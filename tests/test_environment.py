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

import unittest

from biofoundation.core.environment import missing_environment_variables, require_environment


class EnvironmentTest(unittest.TestCase):
    def test_reports_unset_and_placeholder_values(self):
        environ = {"DATA_PATH": "/data", "CHECKPOINT_DIR": "#CHANGEME"}
        self.assertEqual(
            missing_environment_variables(("DATA_PATH", "CHECKPOINT_DIR", "OUTPUT_PATH"), environ),
            ("CHECKPOINT_DIR", "OUTPUT_PATH"),
        )

    def test_accepts_complete_environment(self):
        require_environment(("DATA_PATH", "CHECKPOINT_DIR"), {"DATA_PATH": "/data", "CHECKPOINT_DIR": "/ckpt"})

    def test_raises_one_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "DATA_PATH, CHECKPOINT_DIR"):
            require_environment(("DATA_PATH", "CHECKPOINT_DIR"), {})


if __name__ == "__main__":
    unittest.main()

