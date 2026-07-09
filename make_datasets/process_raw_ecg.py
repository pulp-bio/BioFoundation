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
#* Author:  Marija Zelic                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import wfdb
import numpy as np
import pandas as pd
import pickle
import argparse
import tqdm
import ast
import os
import math

from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from mat73 import loadmat

def preprocess_signal(waveform, fs, low, high, downsample_fs=None, upsample_fs=None):
    """
    Preprocess ECG&PPG waveform. 

    wavefrom (np.array): Numpy array of shape (12x5000).
    band (tuple): Tuple containing band for bandpass filtering. 
    fs (int): Sampling frequency.
    low (int): Low cut frequency for bandpass filter.
    high (int): High cut frequency for bandpass filter.
    downsample_fs (int): Downsampling frequency. If None, no downsampling is applied.
    upsample_fs (int): Upsampling frequency. If None, no upsampling is applied

    """
    # -------- Missing values -----------------
    # At least checking for them
    # For now PTB-XL doesn't have it
    check_nan_inf = (~np.isfinite(waveform)).sum()
    if check_nan_inf != 0:
        # Here we will put some kind of processing at one moment
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
    # ------- Upsampling -----------------
    if upsample_fs is not None:
        waveform = resample_poly(waveform, up=upsample_fs, down=fs, axis=-1)
        fs = upsample_fs
    
    # -------- Bandpass filtering ------------
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    waveform_bp = filtfilt(b, a, waveform, axis=-1)

    # -------- Notch filtering -------------
    notch_freq_1 = 50
    notch_freq_2 = 60
    q = 30 # quality factor?
    
    # First notch filter
    b_notch, a_notch = iirnotch(notch_freq_1, q, fs)
    waveform_notch = filtfilt(b_notch, a_notch, waveform_bp, axis=-1)
    
    # Second notch filter
    b_notch, a_notch = iirnotch(notch_freq_2, q, fs)
    waveform_notch = filtfilt(b_notch, a_notch, waveform_notch, axis=-1)
     
    # -------- Downsampling ----------------
    if downsample_fs is not None:
        # This is general down-sampling pipeline working for any frequency
        # We need to find greatest common divisor of the two to have up and down factors
        gcd_value = math.gcd(fs, downsample_fs)
        up_factor = downsample_fs // gcd_value
        down_factor = fs // gcd_value

        resampled = resample_poly(waveform_notch, up=up_factor, down=down_factor, axis=-1)
    else:
        resampled = waveform_notch

    return resampled

def time_segmenting(signal, split_signal, sampling_rate, downsample_fs=None, upsample_fs=None):
    """
    Segments the data into splits of length split_signal (in seconds).
    Assumes time length of signal si dividable with split_signal, i.e. there's no need for padding.
    downsample_fs and upsample_fs are used to determine effective sampling rate after preprocessing. 
    One of them should be None. 
    
    Args:
        signal (np.array): Processed signal of shape [num_channels, time_length].
        split_signal (int): Length of time segment in seconds for splitting the signal. 
        sampling_rate (int): Original sampling rate of the signal. 
        downsample_fs (int): Downsampling frequency. If None, no downsampling is applied. 
        upsample_fs (int): Upsampling frequency. If None, no upsampling is applied. 
    """
    
    if downsample_fs is not None:
        split_samples = downsample_fs * split_signal
    elif upsample_fs is not None:
        split_samples = upsample_fs * split_signal
    else:
        split_samples = sampling_rate * split_signal
        
    # We assume it's dividable
    assert signal.shape[-1] % split_samples == 0, \
        f"Signal length ({signal.shape[-1]} must be exactly divisible by {split_samples})."
        
    num_sections = signal.shape[-1] // split_samples
    
    # Splitting
    pieces = np.split(signal, num_sections, axis=-1)
    
    return pieces