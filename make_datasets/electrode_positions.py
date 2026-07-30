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
#* Author:  Glenn Anta Bucagu                                                 *
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

"""3D electrode coordinates for the standard 118-electrode layout.

Angular positions follow the BESA electrode and surface location tables at
https://wiki.besa.de/index.php?title=Electrodes_and_Surface_Locations. Reference
electrodes that have no scalp position of their own are given fixed coordinates.
"""

import numpy as np


SPECIAL_REFERENCE_POSITIONS = {
    'AR': (0.0, 0.0, 0.2),  
    'LE':  (0, 0, -0.615),
}

ELECTRODE_ANGLES = {
    'F11': {'theta': -130, 'phi': -40},
    'F9': {'theta': -115, 'phi': -35},
    'F7': {'theta': -92, 'phi': -36},
    'F5': {'theta': -75, 'phi': -41},
    'F3': {'theta': -60, 'phi': -51},
    'F1': {'theta': -50, 'phi': -68},
    'FZ': {'theta': 46, 'phi': 90},
    'F2': {'theta': 50, 'phi': 68},
    'F4': {'theta': 60, 'phi': 51},
    'F6': {'theta': 75, 'phi': 41},
    'F8': {'theta': 92, 'phi': 36},
    'F10': {'theta': 115, 'phi': 35},
    'F12': {'theta': 130, 'phi': 40},
    'FT11': {'theta': -130, 'phi': -22},
    'FT9': {'theta': -115, 'phi': -18},
    'FT7': {'theta': -92, 'phi': -18},
    'FC5': {'theta': -71, 'phi': -21},
    'FC3': {'theta': -50, 'phi': -28},
    'FC1': {'theta': -32, 'phi': -45},
    'FCZ': {'theta': 23, 'phi': 90},
    'FC2': {'theta': 32, 'phi': 45},
    'FC4': {'theta': 50, 'phi': 28},
    'FC6': {'theta': 71, 'phi': 21},
    'FT8': {'theta': 92, 'phi': 18},
    'FT10': {'theta': 115, 'phi': 18},
    'FT12': {'theta': 130, 'phi': 22},
    'T9': {'theta': -115, 'phi': 0},
    'LPA': {'theta': -115, 'phi': 0},
    'T7': {'theta': -92, 'phi': 0},
    'C5': {'theta': -69, 'phi': 0},
    'C3': {'theta': -46, 'phi': 0},
    'C1': {'theta': -23, 'phi': 0},
    'CZ': {'theta': 0, 'phi': 0},
    'C2': {'theta': 23, 'phi': 0},
    'C4': {'theta': 46, 'phi': 0},
    'C6': {'theta': 69, 'phi': 0},
    'T8': {'theta': 92, 'phi': 0},
    'T10': {'theta': 115, 'phi': 0},
    'RPA': {'theta': 115, 'phi': 0},
    'P11': {'theta': -130, 'phi': 40},
    'P9': {'theta': -115, 'phi': 36},
    'P7': {'theta': -92, 'phi': 36},
    'P5': {'theta': -75, 'phi': 41},
    'P3': {'theta': -60, 'phi': 51},
    'P1': {'theta': -50, 'phi': 68},
    'PZ': {'theta': 46, 'phi': -90},
    'P2': {'theta': 50, 'phi': -68},
    'P4': {'theta': 60, 'phi': -51},
    'P6': {'theta': 75, 'phi': -41},
    'P8': {'theta': 92, 'phi': -36},
    'P10': {'theta': 115, 'phi': -36},
    'P12': {'theta': 130, 'phi': -40},
    'TP9': {'theta': -115, 'phi': 18},
    'TP7': {'theta': -92, 'phi': 18},
    'CP5': {'theta': -71, 'phi': 21},
    'CP3': {'theta': -50, 'phi': 28},
    'CP1': {'theta': -32, 'phi': 45},
    'CPZ': {'theta': 23, 'phi': -90},
    'CP2': {'theta': 32, 'phi': -45},
    'CP4': {'theta': 50, 'phi': -28},
    'CP6': {'theta': 71, 'phi': -21},
    'TP8': {'theta': 92, 'phi': -18},
    'TP10': {'theta': 115, 'phi': -18},
    'AF9': {'theta': -115, 'phi': -47},
    'AF7': {'theta': -92, 'phi': -52},
    'AF5': {'theta': -83, 'phi': -59},
    'AF3': {'theta': -74, 'phi': -67},
    'AF1': {'theta': -71, 'phi': -78},
    'AFZ': {'theta': 69, 'phi': 90},
    'AF2': {'theta': 71, 'phi': 78},
    'AF4': {'theta': 74, 'phi': 67},
    'AF6': {'theta': 83, 'phi': 59},
    'AF8': {'theta': 92, 'phi': 52},
    'AF10': {'theta': 115, 'phi': 47},
    'FP1': {'theta': -92, 'phi': -72},
    'FPZ': {'theta': 92, 'phi': 90},
    'FP2': {'theta': 92, 'phi': 72},
    'NZ': {'theta': 112, 'phi': 90},
    'NAS': {'theta': 112, 'phi': 90},
    'Chin': {'theta': 155, 'phi': 90},
    'LO1': {'theta': -118, 'phi': -48},
    'LO2': {'theta': 118, 'phi': 48},
    'SO1': {'theta': -105, 'phi': -65},
    'SO2': {'theta': 105, 'phi': 65},
    'IO1': {'theta': -125, 'phi': -63},
    'IO2': {'theta': 125, 'phi': 63},
    'T3': {'theta': -92, 'phi': 0},
    'T4': {'theta': 92, 'phi': 0},
    'T5': {'theta': -92, 'phi': 36},
    'T6': {'theta': 92, 'phi': -36},
    'A1': {'theta': -128, 'phi': 3},
    'A2': {'theta': 128, 'phi': -3},
    'T1': {'theta': -108, 'phi': -20},
    'T2': {'theta': 108, 'phi': 20},
    'O1': {'theta': -92, 'phi': 72},
    'OZ': {'theta': 92, 'phi': -90},
    'O2': {'theta': 92, 'phi': -72},
    'O9': {'theta': -115, 'phi': 72},
    'O10': {'theta': 115, 'phi': -72},
    'CB1': {'theta': -130, 'phi': 45},
    'CB2': {'theta': 130, 'phi': -45},
    'IZ': {'theta': 115, 'phi': -90},
    'Neck': {'theta': 150, 'phi': -90},
    'SP1': {'theta': -145, 'phi': -25},
    'SP2': {'theta': 145, 'phi': 25},
    'M1': {'theta': -120, 'phi': 25},
    'M2': {'theta': 120, 'phi': -25},
    'PO9': {'theta': -115, 'phi': 54},
    'PO7': {'theta': -92, 'phi': 54},
    'PO5': {'theta': -83, 'phi': 59},
    'PO3': {'theta': -74, 'phi': 67},
    'PO1': {'theta': -71, 'phi': 78},
    'POZ': {'theta': 69, 'phi': -90},
    'PO2': {'theta': 71, 'phi': -78},
    'PO4': {'theta': 74, 'phi': -67},
    'PO6': {'theta': 83, 'phi': -59},
    'PO8': {'theta': 92, 'phi': -54},
    'PO10': {'theta': 115, 'phi': -54}
}


def get_electrode_3d_positions(theta, phi, radius=1):
    """
    Converts spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z).
    We use radius = 1 to have to normalize each coordinate in the range [0, 1]

    Args:
        theta (float): Azimuthal angle in degrees.
        phi (float): Polar angle in degrees.
        radius (float): Radius of the sphere.

    Returns:
        tuple: (x, y, z) 3D Cartesian coordinates.
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    x = radius * np.cos(phi_rad) * np.cos(theta_rad)
    y = radius * np.cos(phi_rad) * np.sin(theta_rad)
    z = radius * np.sin(phi_rad)

    return x, y, z
