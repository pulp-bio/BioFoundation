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

from biofoundation.core.batch import BatchRequirements, as_signal_batch, require_batch_fields


class SignalBatchTest(unittest.TestCase):
    def test_normalizes_raw_input(self):
        signal = object()
        self.assertIs(as_signal_batch(signal)["input"], signal)

    def test_normalizes_input_label_tuple(self):
        signal = object()
        label = object()
        batch = as_signal_batch((signal, label))
        self.assertIs(batch["input"], signal)
        self.assertIs(batch["label"], label)

    def test_preserves_mapping_metadata(self):
        source = {"input": object(), "channel_locations": object(), "sensor_type": object()}
        batch = as_signal_batch(source)
        self.assertEqual(batch, source)
        self.assertIsNot(batch, source)

    def test_rejects_ambiguous_sequences(self):
        with self.assertRaisesRegex(ValueError, "received 3 values"):
            as_signal_batch((1, 2, 3))

    def test_rejects_mapping_without_input(self):
        with self.assertRaisesRegex(ValueError, "'input' field"):
            as_signal_batch({"label": 1})

    def test_validates_model_specific_metadata(self):
        requirements = BatchRequirements(channel_locations=True, sensor_type=True)
        batch = as_signal_batch({"input": object(), "channel_locations": object()})
        with self.assertRaisesRegex(ValueError, "sensor_type"):
            require_batch_fields(batch, requirements)

        batch["sensor_type"] = object()
        self.assertIs(require_batch_fields(batch, requirements), batch)

    def test_validates_paired_electrode_geometry_and_padding(self):
        requirements = BatchRequirements(channel_coords=True, num_padded_channels=True)
        batch = as_signal_batch({"input": object(), "channel_coords": object()})
        with self.assertRaisesRegex(ValueError, "num_padded_channels"):
            require_batch_fields(batch, requirements)

        batch["num_padded_channels"] = object()
        self.assertIs(require_batch_fields(batch, requirements), batch)

    def test_geometry_representations_are_independent(self):
        """A model requiring one geometry field is not satisfied by the other."""

        coords_only = as_signal_batch({"input": object(), "channel_coords": object()})
        with self.assertRaisesRegex(ValueError, "channel_locations"):
            require_batch_fields(coords_only, BatchRequirements(channel_locations=True))

        midpoints_only = as_signal_batch({"input": object(), "channel_locations": object()})
        with self.assertRaisesRegex(ValueError, "channel_coords"):
            require_batch_fields(midpoints_only, BatchRequirements(channel_coords=True))

    def test_defaults_leave_existing_requirements_unchanged(self):
        """Fields added for newer families must not alter an existing spec's equality."""

        self.assertEqual(BatchRequirements(), BatchRequirements())
        self.assertEqual(
            BatchRequirements(channel_locations=True),
            BatchRequirements(channel_locations=True),
        )
        self.assertNotEqual(
            BatchRequirements(channel_locations=True),
            BatchRequirements(channel_coords=True),
        )


if __name__ == "__main__":
    unittest.main()

