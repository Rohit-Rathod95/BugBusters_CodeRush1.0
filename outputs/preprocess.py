"""
EEG Data Preprocessing Script
Handles raw .edf files and applies standard preprocessing steps.
"""

import argparse
import os
import glob
import warnings
import mne
from mne.preprocessing import ICA
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


def preprocess_file(path, out_dir, sr=128, l_freq=1.0, h_freq=40.0, notch_freqs=(50,)):
    """
    Preprocess a single EEG file.
    
    Args:
        path (str): Path to input .edf file
        out_dir (str): Output directory for processed files
        sr (int): Target sampling rate
        l_freq (float): Low-pass filter frequency
        h_freq (float): High-pass filter frequency
        notch_freqs (tuple): Notch filter frequencies
    
    Returns:
        str: Path to output file
    """
    try:
        # Load raw data
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        
        # Pick EEG and EOG channels only
        raw.pick_types(eeg=True, eog=True, misc=False, stim=False, exclude='bads')
        
        # Set montage if channels match standard 10-20
        try:
            raw.set_montage('standard_1020', match_case=False, verbose=False)
        except Exception:
            print(f"Warning: Could not set montage for {os.path.basename(path)}")
        
        # Apply filters
        raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        raw.notch_filter(notch_freqs, fir_design='firwin', verbose=False)
        
        # Resample
        if raw.info['sfreq'] != sr:
            raw.resample(sr, npad="auto", verbose=False)
        
        # ICA for artifact removal
        try:
            n_components = min(20, int(len(raw.ch_names) * 0.8))
            if n_components < 5:
                n_components = min(len(raw.ch_names), 10)
            
            ica = ICA(
                n_components=n_components, 
                random_state=97, 
                max_iter=500,
                verbose=False
            )
            ica.fit(raw, verbose=False)
            
            # Find and exclude EOG artifacts
            eog_inds, scores = ica.find_bads_eog(raw, threshold=2.5, verbose=False)
            ica.exclude = list(eog_inds)
            raw = ica.apply(raw.copy(), verbose=False)
            
        except Exception as e:
            print(f"ICA failed for {os.path.basename(path)}: {str(e)[:50]}...")
        
        # Save processed file
        base = os.path.splitext(os.path.basename(path))[0]
        outpath = os.path.join(out_dir, base + '_clean.fif')
        raw.save(outpath, overwrite=True, verbose=False)
        
        return outpath
        
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG .edf files')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Directory containing .edf files (can contain subdirs like healthy/affected)')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for processed .fif files')
    parser.add_argument('--sr', type=int, default=128,
                       help='Target sampling rate (default: 128)')
    parser.add_argument('--l_freq', type=float, default=1.0,
                       help='Low-pass filter frequency (default: 1.0)')
    parser.add_argument('--h_freq', type=float, default=40.0,
                       help='High-pass filter frequency (default: 40.0)')
    parser.add_argument('--notch_freq', type=float, default=50.0,
                       help='Notch filter frequency (default: 50.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .edf files recursively
    pattern = os.path.join(args.input_dir, '**', '*.edf')
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"No .edf files found in {args.input_dir}")
        print("Please check the directory path and ensure it contains .edf files")
        return
    
    print(f"Found {len(files)} .edf files")
    print(f"Processing with parameters:")
    print(f"  - Sampling rate: {args.sr} Hz")
    print(f"  - Bandpass filter: {args.l_freq}-{args.h_freq} Hz")
    print(f"  - Notch filter: {args.notch_freq} Hz")
    print(f"  - Output directory: {args.output_dir}")
    
    # Process files
    successful = 0
    failed = 0
    
    for file_path in tqdm(files, desc='Preprocessing files'):
        result = preprocess_file(
            file_path, 
            args.output_dir,
            sr=args.sr,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch_freqs=(args.notch_freq,)
        )
        
        if result:
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  - Successfully processed: {successful} files")
    print(f"  - Failed: {failed} files")
    print(f"  - Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()