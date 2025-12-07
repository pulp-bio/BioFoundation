# *----------------------------------------------------------------------------*
# * Copyright (C) 2025 ETH Zurich, Switzerland                                 *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Fasulo                                                     *
# *----------------------------------------------------------------------------*
import argparse

import torch
from safetensors.torch import save_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch Lightning checkpoint to a safetensors file."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the PyTorch Lightning checkpoint file.",
    )
    parser.add_argument(
        "--safetensor_path",
        type=str,
        default="model.safetensors",
        help="Path to save the converted safetensors file.",
    )
    parser.add_argument(
        "--exclude_keys",
        type=str,
        nargs="*",
        default=[],
        help="List of keys to exclude from the safetensors file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print the keys of the parameters being saved.",
    )
    args = parser.parse_args()

    # Load the PyTorch Lightning checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    # Extract the model state_dict and filter out excluded keys if any
    parameters = {
        k: v for k, v in ckpt["state_dict"].items() if k not in args.exclude_keys
    }

    # Options for verbose output - list the keys being saved
    if args.verbose:
        print("The following keys will be saved in the safetensors file:")
        for key in parameters.keys():
            print(f" - {key}")

    # Save the parameters in safetensors format
    save_file(parameters, args.safetensor_path)
    print(f"Safetensors file saved to {args.safetensor_path}")
