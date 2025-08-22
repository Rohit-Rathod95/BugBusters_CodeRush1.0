#!/usr/bin/env python3
"""
Test script for EEG preprocessing pipeline

This script generates sample EEG data to test the preprocessing functionality.
"""

import numpy as np
import mne
import os
from pathlib import Path

def create_sample_eeg_data():
    """Create sample EEG data for testing."""
    
    # Create sample data directory
    test_dir = "test_data"
    Path(test_dir).mkdir(exist_ok=True)
    Path(os.path.join(test_dir, "affected")).mkdir(exist_ok=True)
    Path(os.path.join(test_dir, "healthy")).mkdir(exist_ok=True)
    
    # EEG parameters
    n_channels = 19  # Standard 10-20 system
    duration = 60  # 60 seconds
    sfreq = 256  # 256 Hz sampling rate
    
    # Channel names (standard 10-20 system)
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
        'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
    ]
    
    # Create sample data for affected group
    for i in range(3):  # Create 3 sample files
        # Generate synthetic EEG-like data with some artifacts
        n_samples = int(duration * sfreq)
        data = np.random.randn(n_channels, n_samples) * 50  # 50 microvolts
        
        # Add some realistic EEG characteristics
        # Alpha rhythm (8-13 Hz) in occipital channels
        t = np.linspace(0, duration, n_samples)
        alpha_freq = 10  # Hz
        alpha_signal = 20 * np.sin(2 * np.pi * alpha_freq * t)
        data[17, :] += alpha_signal  # O1
        data[18, :] += alpha_signal  # O2
        
        # Beta rhythm (13-30 Hz) in frontal channels
        beta_freq = 20  # Hz
        beta_signal = 15 * np.sin(2 * np.pi * beta_freq * t)
        data[0, :] += beta_signal   # Fp1
        data[1, :] += beta_signal   # Fp2
        
        # Add some noise and artifacts
        data += np.random.randn(n_channels, n_samples) * 10
        
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(data, info)
        
        # Save as FIF file
        output_path = os.path.join(test_dir, "affected", f"sample_affected_{i+1}.fif")
        raw.save(output_path, overwrite=True)
        print(f"Created: {output_path}")
    
    # Create sample data for healthy group
    for i in range(3):  # Create 3 sample files
        # Generate synthetic EEG-like data (slightly different characteristics)
        n_samples = int(duration * sfreq)
        data = np.random.randn(n_channels, n_samples) * 40  # 40 microvolts (less variance)
        
        # Add some realistic EEG characteristics
        t = np.linspace(0, duration, n_samples)
        
        # Theta rhythm (4-7 Hz) in central channels
        theta_freq = 6  # Hz
        theta_signal = 25 * np.sin(2 * np.pi * theta_freq * t)
        data[8, :] += theta_signal   # C3
        data[9, :] += theta_signal   # Cz
        data[10, :] += theta_signal  # C4
        
        # Delta rhythm (0.5-4 Hz) in frontal channels
        delta_freq = 2  # Hz
        delta_signal = 30 * np.sin(2 * np.pi * delta_freq * t)
        data[3, :] += delta_signal   # F3
        data[4, :] += delta_signal   # Fz
        data[5, :] += delta_signal   # F4
        
        # Add some noise
        data += np.random.randn(n_channels, n_samples) * 8
        
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(data, info)
        
        # Save as FIF file
        output_path = os.path.join(test_dir, "healthy", f"sample_healthy_{i+1}.fif")
        raw.save(output_path, overwrite=True)
        print(f"Created: {output_path}")
    
    print(f"\nSample EEG data created in '{test_dir}' directory")
    print("You can now test the preprocessing script with:")
    print(f"python preprocess.py --input_dir {test_dir} --output_dir data/clean --sr 128")

if __name__ == "__main__":
    create_sample_eeg_data()
