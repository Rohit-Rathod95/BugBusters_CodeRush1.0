"""
Feature Extraction Script
Extracts scalogram images and metadata features from preprocessed EEG data.
"""

import argparse
import os
import glob
import warnings
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from PIL import Image
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')  # Use non-interactive backend


class FeatureExtractor:
    def __init__(self, epoch_length=2.0, overlap=0.5, target_sfreq=128):
        """
        Initialize feature extractor.
        
        Args:
            epoch_length (float): Length of each epoch in seconds
            overlap (float): Overlap ratio between epochs (0-1)
            target_sfreq (int): Target sampling frequency
        """
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.target_sfreq = target_sfreq
        
        # EEG frequency bands
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
    
    def create_scalogram(self, data, sfreq):
        """
        Create CWT scalogram from EEG data.
        
        Args:
            data (np.array): EEG data (channels x time)
            sfreq (float): Sampling frequency
        
        Returns:
            np.array: 224x224 scalogram image
        """
        # Average across channels for multi-channel data
        if data.ndim > 1:
            signal_data = np.mean(data, axis=0)
        else:
            signal_data = data
        
        # Define frequency range and scales
        freq_min, freq_max = 1, 40
        num_scales = 224
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_scales)
        scales = pywt.frequency2scale('cmor1.5-1.0', frequencies / sfreq)
        
        # Compute CWT
        coefficients, _ = pywt.cwt(signal_data, scales, 'cmor1.5-1.0')
        
        # Take magnitude and apply log transform
        scalogram = np.abs(coefficients)
        scalogram = np.log10(scalogram + 1e-10)
        
        # Resize to 224x224
        if scalogram.shape[1] != 224:
            # Interpolate time dimension to 224 points
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, scalogram.shape[1])
            x_new = np.linspace(0, 1, 224)
            f = interp1d(x_old, scalogram, axis=1, kind='linear')
            scalogram = f(x_new)
        
        # Normalize to [0, 255]
        scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())
        scalogram = (scalogram * 255).astype(np.uint8)
        
        return scalogram
    
    def compute_psd_features(self, data, sfreq):
        """
        Compute power spectral density features for EEG bands.
        
        Args:
            data (np.array): EEG data (channels x time)
            sfreq (float): Sampling frequency
        
        Returns:
            dict: PSD features for each frequency band
        """
        features = {}
        
        # Average across channels
        if data.ndim > 1:
            signal_data = np.mean(data, axis=0)
        else:
            signal_data = data
        
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(signal_data, sfreq, nperseg=min(len(signal_data), sfreq))
        
        # Extract power for each frequency band
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                features[f'{band_name}_power'] = band_power
            else:
                features[f'{band_name}_power'] = 0.0
        
        # Compute relative powers
        total_power = sum(features.values())
        if total_power > 0:
            for band_name in self.freq_bands.keys():
                features[f'{band_name}_rel_power'] = features[f'{band_name}_power'] / total_power
        else:
            for band_name in self.freq_bands.keys():
                features[f'{band_name}_rel_power'] = 0.0
        
        return features
    
    def extract_epochs(self, raw):
        """
        Extract epochs from raw EEG data using sliding window.
        
        Args:
            raw (mne.Raw): Raw EEG data
        
        Returns:
            list: List of epoch data arrays
        """
        sfreq = raw.info['sfreq']
        n_samples = int(self.epoch_length * sfreq)
        step_samples = int(n_samples * (1 - self.overlap))
        
        data = raw.get_data()
        epochs = []
        
        for start in range(0, data.shape[1] - n_samples + 1, step_samples):
            end = start + n_samples
            epoch_data = data[:, start:end]
            epochs.append(epoch_data)
        
        return epochs
    
    def process_file(self, file_path, output_dir):
        """
        Process a single .fif file and extract features.
        
        Args:
            file_path (str): Path to .fif file
            output_dir (str): Output directory for images
        
        Returns:
            list: List of feature dictionaries
        """
        try:
            # Load preprocessed data
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            
            # Extract label from file path
            if 'healthy' in file_path.lower():
                label = 0
            elif 'affected' in file_path.lower():
                label = 1
            else:
                # Try to infer from parent directory
                parent_dir = os.path.basename(os.path.dirname(file_path))
                if 'healthy' in parent_dir.lower():
                    label = 0
                elif 'affected' in parent_dir.lower():
                    label = 1
                else:
                    print(f"Warning: Cannot determine label for {file_path}")
                    label = -1
            
            # Extract epochs
            epochs = self.extract_epochs(raw)
            
            features_list = []
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            for i, epoch_data in enumerate(epochs):
                # Create scalogram
                scalogram = self.create_scalogram(epoch_data, raw.info['sfreq'])
                
                # Save scalogram image
                img_name = f"{base_name}_epoch_{i:03d}.png"
                img_path = os.path.join(output_dir, img_name)
                
                # Convert to RGB image
                scalogram_rgb = np.stack([scalogram, scalogram, scalogram], axis=-1)
                img = Image.fromarray(scalogram_rgb)
                img.save(img_path)
                
                # Compute PSD features
                psd_features = self.compute_psd_features(epoch_data, raw.info['sfreq'])
                
                # Combine features
                features = {
                    'file_path': file_path,
                    'image_path': img_path,
                    'epoch_idx': i,
                    'label': label,
                    **psd_features
                }
                
                features_list.append(features)
            
            return features_list
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []


def compute_cohort_stats(features_df, output_dir):
    """
    Compute and save mean bandpowers for healthy and affected cohorts.
    
    Args:
        features_df (pd.DataFrame): Features dataframe
        output_dir (str): Output directory
    """
    band_cols = [col for col in features_df.columns if '_power' in col and '_rel_' not in col]
    
    cohort_stats = {}
    for label, group in features_df.groupby('label'):
        label_name = 'healthy' if label == 0 else 'affected'
        cohort_stats[label_name] = {
            'mean': group[band_cols].mean().to_dict(),
            'std': group[band_cols].std().to_dict(),
            'n_samples': len(group)
        }
    
    # Save cohort statistics
    stats_path = os.path.join(output_dir, 'cohort_stats.pkl')
    joblib.dump(cohort_stats, stats_path)
    
    # Save as CSV for human readability
    stats_df = pd.DataFrame({
        'band': [col.replace('_power', '') for col in band_cols]
    })
    
    for label_name in cohort_stats.keys():
        stats_df[f'{label_name}_mean'] = [cohort_stats[label_name]['mean'][col] for col in band_cols]
        stats_df[f'{label_name}_std'] = [cohort_stats[label_name]['std'][col] for col in band_cols]
    
    stats_csv_path = os.path.join(output_dir, 'cohort_stats.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    
    print(f"Cohort statistics saved to {stats_path} and {stats_csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from preprocessed EEG data')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Directory containing preprocessed .fif files')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for features and images')
    parser.add_argument('--epoch_length', type=float, default=2.0,
                       help='Epoch length in seconds (default: 2.0)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio between epochs (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Find all .fif files
    pattern = os.path.join(args.input_dir, '**', '*.fif')
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"No .fif files found in {args.input_dir}")
        return
    
    print(f"Found {len(files)} .fif files")
    print(f"Extracting features with:")
    print(f"  - Epoch length: {args.epoch_length} seconds")
    print(f"  - Overlap: {args.overlap * 100}%")
    print(f"  - Output directory: {args.output_dir}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        epoch_length=args.epoch_length,
        overlap=args.overlap
    )
    
    # Process all files
    all_features = []
    
    for file_path in tqdm(files, desc='Extracting features'):
        features = extractor.process_file(file_path, images_dir)
        all_features.extend(features)
    
    if not all_features:
        print("No features extracted. Please check input files.")
        return
    
    # Create features dataframe
    features_df = pd.DataFrame(all_features)
    
    # Save features CSV
    features_csv_path = os.path.join(args.output_dir, 'features.csv')
    features_df.to_csv(features_csv_path, index=False)
    
    # Compute and save cohort statistics
    if features_df['label'].nunique() > 1:
        compute_cohort_stats(features_df, args.output_dir)
    
    print(f"\nFeature extraction complete:")
    print(f"  - Total epochs: {len(features_df)}")
    print(f"  - Features CSV: {features_csv_path}")
    print(f"  - Images directory: {images_dir}")
    
    # Print label distribution
    label_counts = features_df['label'].value_counts()
    for label, count in label_counts.items():
        label_name = 'healthy' if label == 0 else 'affected' if label == 1 else 'unknown'
        print(f"  - {label_name}: {count} epochs")


if __name__ == "__main__":
    main()