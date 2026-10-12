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
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

"""Extract the ISRUC-Sleep subgroup I archives into the layout make_isruc.py reads.

ISRUC is distributed as ``1.rar`` to ``100.rar``, each unpacking to
``{subject}/{subject}.rec`` with its hypnograms. This needs the ``rarfile`` package and
an extraction tool it can drive, such as ``unrar``, ``unar`` or ``bsdtar``. Extracting
the archives with any of those tools directly produces the same layout.
"""

import argparse
import os
import sys


def main():
    """Extract every numbered archive found in the input directory."""
    parser = argparse.ArgumentParser(description="Extract the ISRUC-Sleep subgroup I archives")
    parser.add_argument("--input_dir", required=True, help="Directory holding 1.rar to 100.rar")
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to extract into; defaults to --input_dir, which make_isruc.py then reads",
    )
    parser.add_argument("--start", type=int, default=1, help="First archive number")
    parser.add_argument("--end", type=int, default=100, help="Last archive number")
    args = parser.parse_args()

    try:
        import rarfile
    except ImportError:
        sys.exit("unrar_isruc.py needs the rarfile package: pip install rarfile")

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    extracted, failed, missing = 0, [], []
    for number in range(args.start, args.end + 1):
        archive = os.path.join(args.input_dir, f"{number}.rar")
        if not os.path.isfile(archive):
            missing.append(number)
            continue
        try:
            with rarfile.RarFile(archive) as handle:
                handle.extractall(output_dir)
            extracted += 1
        except (rarfile.Error, OSError) as error:
            failed.append(f"{number}.rar ({error})")

    print(f"Extracted {extracted} archives into {output_dir}")
    if missing:
        print(f"  {len(missing)} archives not found, e.g. {missing[:5]}")
    if failed:
        print(f"  {len(failed)} archives failed: " + "; ".join(failed[:5]))
        sys.exit(1)


if __name__ == "__main__":
    main()
