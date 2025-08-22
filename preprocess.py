#!/usr/bin/env python3
"""
EEG Data Preprocessing Script for Schizophrenia Detection

This script preprocesses raw EEG data by:
1. Loading EEG files from the input directory
2. Applying bandpass filtering (0.5-45 Hz)
3. Resampling to specified sampling rate
4. Saving preprocessed data to output directory

Usage:
    python preprocess.py --input_dir data/raw --output_dir data/clean --sr 128
"""

import argparse
import os
import glob
import numpy as np
import mne
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(output_dir):
    """Create output directories if they don't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'affected')).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, 'healthy')).mkdir(exist_ok=True)

def load_eeg_file(file_path):
    """Load EEG file using MNE."""
    try:
        # Try to load as various formats
        if file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.bdf'):
            raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.gdf'):
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.set'):
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return None
        
        return raw
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def preprocess_eeg(raw, target_sr=128):
    """Preprocess EEG data with filtering and resampling."""
    try:
        # Apply bandpass filter (0.5-45 Hz)
        raw.filter(0.5, 45, method='iir', verbose=False)
        
        # Resample to target sampling rate
        if raw.info['sfreq'] != target_sr:
            raw.resample(target_sr, verbose=False)
        
        # Get the data as numpy array
        data = raw.get_data()
        
        return data, raw.info
        
    except Exception as e:
        logger.error(f"Error preprocessing EEG data: {e}")
        return None, None

def save_preprocessed_data(data, info, output_path):
    """Save preprocessed data as numpy array."""
    try:
        # Save data and metadata
        np.savez_compressed(
            output_path,
            data=data,
            sfreq=info['sfreq'],
            ch_names=info['ch_names'],
            n_channels=info['nchan'],
            n_samples=data.shape[1]
        )
        return True
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        return False

def process_directory(input_dir, output_dir, target_sr):
    """Process all EEG files in the input directory."""
    setup_directories(output_dir)
    
    # Process affected and healthy directories
    for condition in ['affected', 'healthy']:
        input_condition_dir = os.path.join(input_dir, condition)
        output_condition_dir = os.path.join(output_dir, condition)
        
        if not os.path.exists(input_condition_dir):
            logger.warning(f"Input directory {input_condition_dir} does not exist")
            continue
        
        # Find all EEG files
        eeg_files = []
        for ext in ['*.edf', '*.bdf', '*.gdf', '*.set']:
            eeg_files.extend(glob.glob(os.path.join(input_condition_dir, ext)))
        
        if not eeg_files:
            logger.warning(f"No EEG files found in {input_condition_dir}")
            continue
        
        logger.info(f"Processing {len(eeg_files)} files in {condition} directory")
        
        for file_path in tqdm(eeg_files, desc=f"Processing {condition}"):
            # Load EEG file
            raw = load_eeg_file(file_path)
            if raw is None:
                continue
            
            # Preprocess data
            data, info = preprocess_eeg(raw, target_sr)
            if data is None:
                continue
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_condition_dir, f"{base_name}_preprocessed.npz")
            
            # Save preprocessed data
            if save_preprocessed_data(data, info, output_path):
                logger.info(f"Successfully processed: {base_name}")
            else:
                logger.error(f"Failed to save: {base_name}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG data for schizophrenia detection')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing raw EEG data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for preprocessed data')
    parser.add_argument('--sr', type=int, default=128, help='Target sampling rate (default: 128 Hz)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory {args.input_dir} does not exist")
        return
    
    logger.info(f"Starting EEG preprocessing...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target sampling rate: {args.sr} Hz")
    
    # Process the data
    process_directory(args.input_dir, args.output_dir, args.sr)
    
    logger.info("EEG preprocessing completed!")

if __name__ == "__main__":
    main()
