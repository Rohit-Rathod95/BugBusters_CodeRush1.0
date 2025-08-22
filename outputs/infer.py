"""
Inference Script for EEG Classification
Performs inference on new .edf files and generates reports.
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
from models import ResNetMetaClassifier, SimpleEEGClassifier
from extract_features import FeatureExtractor
import tempfile
from datetime import datetime
import json

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
plt.style.use('seaborn-v0_8')


class EEGInference:
    """
    Inference system for EEG classification.
    """
    
    def __init__(self, model_path, cohort_stats_path=None):
        """
        Initialize inference system.
        
        Args:
            model_path (str): Path to trained model
            cohort_stats_path (str): Path to cohort statistics (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_config = None
        self.cohort_stats = None
        
        # Load model
        self.load_model(model_path)
        
        # Load cohort statistics if available
        if cohort_stats_path and os.path.exists(cohort_stats_path):
            self.cohort_stats = joblib.load(cohort_stats_path)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
    
    def load_model(self, model_path):
        """Load trained model."""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_class = checkpoint.get('model_class', 'ResNetMetaClassifier')
        self.model_config = checkpoint.get('config', {})
        
        # Create model based on class
        if model_class == 'ResNetMetaClassifier':
            self.model = ResNetMetaClassifier(
                n_metadata_features=self.model_config.get('n_metadata_features', 10),
                dropout_rate=self.model_config.get('dropout_rate', 0.5)
            )
        else:  # SimpleEEGClassifier
            self.model = SimpleEEGClassifier(
                n_features=self.model_config.get('n_metadata_features', 10),
                dropout_rate=self.model_config.get('dropout_rate', 0.5)
            )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {model_class}")
        print(f"Device: {self.device}")
    
    def preprocess_edf(self, edf_path, temp_dir):
        """
        Preprocess a single EDF file.
        
        Args:
            edf_path (str): Path to EDF file
            temp_dir (str): Temporary directory for processing
        
        Returns:
            mne.Raw: Preprocessed raw data
        """
        try:
            # Load and preprocess similar to preprocess.py
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.pick_types(eeg=True, eog=True, misc=False, stim=False, exclude='bads')
            
            # Set montage
            try:
                raw.set_montage('standard_1020', match_case=False, verbose=False)
            except:
                pass
            
            # Apply filters
            raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
            raw.notch_filter(50.0, fir_design='firwin', verbose=False)
            
            # Resample
            if raw.info['sfreq'] != 128:
                raw.resample(128, npad="auto", verbose=False)
            
            # Basic artifact removal (simplified)
            try:
                from mne.preprocessing import ICA
                n_components = min(10, int(len(raw.ch_names) * 0.8))
                if n_components >= 3:
                    ica = ICA(n_components=n_components, random_state=97, max_iter=200, verbose=False)
                    ica.fit(raw, verbose=False)
                    eog_inds, _ = ica.find_bads_eog(raw, threshold=2.5, verbose=False)
                    if eog_inds:
                        ica.exclude = list(eog_inds[:2])  # Limit to 2 components
                        raw = ica.apply(raw.copy(), verbose=False)
            except:
                pass
            
            return raw
            
        except Exception as e:
            print(f"Error preprocessing {edf_path}: {e}")
            return None
    
    def extract_features_from_raw(self, raw, temp_dir):
        """
        Extract features from preprocessed raw data.
        
        Args:
            raw (mne.Raw): Preprocessed raw data
            temp_dir (str): Temporary directory for images
        
        Returns:
            tuple: (scalogram_paths, metadata_features)
        """
        # Extract epochs
        epochs = self.feature_extractor.extract_epochs(raw)
        
        scalogram_paths = []
        metadata_features = []
        
        for i, epoch_data in enumerate(epochs):
            # Create scalogram
            scalogram = self.feature_extractor.create_scalogram(epoch_data, raw.info['sfreq'])
            
            # Save scalogram image
            img_name = f"temp_epoch_{i:03d}.png"
            img_path = os.path.join(temp_dir, img_name)
            
            # Convert to RGB and save
            scalogram_rgb = np.stack([scalogram, scalogram, scalogram], axis=-1)
            img = Image.fromarray(scalogram_rgb)
            img.save(img_path)
            scalogram_paths.append(img_path)
            
            # Extract metadata features
            psd_features = self.feature_extractor.compute_psd_features(epoch_data, raw.info['sfreq'])
            metadata_features.append(psd_features)
        
        return scalogram_paths, metadata_features
    
    def predict_single_file(self, edf_path):
        """
        Predict on a single EDF file.
        
        Args:
            edf_path (str): Path to EDF file
        
        Returns:
            dict: Prediction results
        """
        print(f"Processing: {os.path.basename(edf_path)}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Preprocess EDF
            raw = self.preprocess_edf(edf_path, temp_dir)
            if raw is None:
                return {'error': 'Failed to preprocess EDF file'}
            
            # Extract features
            scalogram_paths, metadata_features = self.extract_features_from_raw(raw, temp_dir)
            
            if not scalogram_paths or not metadata_features:
                return {'error': 'Failed to extract features'}
            
            # Make predictions on all epochs
            predictions = []
            
            with torch.no_grad():
                for img_path, meta_feat in zip(scalogram_paths, metadata_features):
                    # Load and preprocess image
                    try:
                        image = Image.open(img_path).convert('RGB')
                        
                        # Transform image
                        import torchvision.transforms as transforms
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
                        ])
                        
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        
                        # Prepare metadata
                        meta_values = [meta_feat.get(key, 0.0) for key in [
                            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
                            'delta_rel_power', 'theta_rel_power', 'alpha_rel_power', 'beta_rel_power', 'gamma_rel_power'
                        ]]
                        
                        meta_tensor = torch.tensor([meta_values], dtype=torch.float32).to(self.device)
                        
                        # Make prediction
                        if isinstance(self.model, ResNetMetaClassifier):
                            output = self.model(image_tensor, meta_tensor)
                        else:
                            output = self.model(meta_tensor)
                        
                        prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                        predictions.append(prob)
                        
                    except Exception as e:
                        print(f"Error processing epoch: {e}")
                        continue
            
            if not predictions:
                return {'error': 'No valid predictions generated'}
            
            # Aggregate predictions
            mean_prob = np.mean(predictions)
            std_prob = np.std(predictions)
            max_prob = np.max(predictions)
            
            # Determine severity
            severity = self.get_severity_level(mean_prob)
            
            # Prepare results
            results = {
                'file_path': edf_path,
                'probability_affected': mean_prob,
                'probability_std': std_prob,
                'max_probability': max_prob,
                'severity': severity,
                'n_epochs': len(predictions),
                'individual_predictions': predictions,
                'metadata_features': metadata_features[0] if metadata_features else {},
                'success': True
            }
            
            return results
    
    def get_severity_level(self, probability):
        """
        Determine severity level based on probability.
        
        Args:
            probability (float): Probability of being affected
        
        Returns:
            str: Severity level
        """
        if probability < 0.2:
            return "Normal"
        elif probability < 0.4:
            return "Mild"
        elif probability < 0.6:
            return "Moderate"
        elif probability < 0.8:
            return "Severe"
        else:
            return "Critical"
    
    def predict_batch(self, input_paths):
        """
        Predict on multiple EDF files.
        
        Args:
            input_paths (list): List of EDF file paths
        
        Returns:
            list: List of prediction results
        """
        results = []
        
        for edf_path in input_paths:
            if os.path.exists(edf_path) and edf_path.endswith('.edf'):
                result = self.predict_single_file(edf_path)
                results.append(result)
            else:
                results.append({
                    'file_path': edf_path,
                    'error': 'File not found or not an EDF file',
                    'success': False
                })
        
        return results
    
    def create_visualization(self, results, output_dir):
        """
        Create visualizations for the results.
        
        Args:
            results (list): List of prediction results
            output_dir (str): Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            print("No successful predictions to visualize")
            return
        
        # Extract data for visualization
        probabilities = [r['probability_affected'] for r in successful_results]
        severities = [r['severity'] for r in successful_results]
        filenames = [os.path.basename(r['file_path']) for r in successful_results]
        
        # 1. Probability distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Probability of Being Affected')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Probabilities')
        plt.grid(True, alpha=0.3)
        
        # 2. Severity distribution
        plt.subplot(2, 2, 2)
        severity_counts = pd.Series(severities).value_counts()
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        severity_order = ['Normal', 'Mild', 'Moderate', 'Severe', 'Critical']
        severity_counts = severity_counts.reindex(severity_order, fill_value=0)
        
        bars = plt.bar(severity_counts.index, severity_counts.values, 
                      color=[colors[i] for i in range(len(severity_counts))])
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        plt.title('Distribution of Severity Levels')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, severity_counts.values):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # 3. Individual predictions
        plt.subplot(2, 2, 3)
        y_pos = np.arange(len(probabilities))
        colors_prob = ['red' if p > 0.5 else 'blue' for p in probabilities]
        
        plt.barh(y_pos, probabilities, color=colors_prob, alpha=0.7)
        plt.yticks(y_pos, [f[:15] + '...' if len(f) > 15 else f for f in filenames])
        plt.xlabel('Probability of Being Affected')
        plt.title('Individual File Predictions')
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        plt.legend()
        
        # 4. Summary statistics
        plt.subplot(2, 2, 4)
        stats_text = f"""
        Summary Statistics:
        
        Total Files: {len(successful_results)}
        Mean Probability: {np.mean(probabilities):.3f}
        Std Probability: {np.std(probabilities):.3f}
        
        Severity Distribution:
        {severity_counts.to_string()}
        
        High Risk Files: {sum(1 for p in probabilities if p > 0.5)}
        """
        
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {os.path.join(output_dir, 'prediction_summary.png')}")
    
    def generate_report(self, results, output_path):
        """
        Generate a detailed report of the predictions.
        
        Args:
            results (list): List of prediction results
            output_path (str): Path to save the report
        """
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_config': self.model_config,
            'summary': {
                'total_files': len(results),
                'successful_predictions': len(successful_results),
                'failed_predictions': len(failed_results),
                'mean_probability': np.mean([r['probability_affected'] for r in successful_results]) if successful_results else 0.0,
                'std_probability': np.std([r['probability_affected'] for r in successful_results]) if successful_results else 0.0,
            },
            'severity_distribution': {},
            'detailed_results': results
        }
        
        # Calculate severity distribution
        if successful_results:
            severities = [r['severity'] for r in successful_results]
            severity_counts = pd.Series(severities).value_counts().to_dict()
            report['severity_distribution'] = severity_counts
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Total files processed: {len(results)}")
        print(f"Successful predictions: {len(successful_results)}")
        print(f"Failed predictions: {len(failed_results)}")
        
        if successful_results:
            print(f"Mean probability: {report['summary']['mean_probability']:.3f}")
            print(f"Std probability: {report['summary']['std_probability']:.3f}")
            print("\nSeverity distribution:")
            for severity, count in report['severity_distribution'].items():
                print(f"  {severity}: {count}")
        
        if failed_results:
            print(f"\nFailed files:")
            for result in failed_results:
                print(f"  {os.path.basename(result['file_path'])}: {result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description='EEG Classification Inference')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Path to EDF file or directory of EDF files')
    parser.add_argument('--output', default='./inference_results', help='Output directory for results')
    parser.add_argument('--cohort-stats', help='Path to cohort statistics file (optional)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize inference system
    print("Initializing EEG Inference System...")
    inference = EEGInference(args.model, args.cohort_stats)
    
    # Collect input files
    input_files = []
    if os.path.isfile(args.input):
        if args.input.endswith('.edf'):
            input_files.append(args.input)
        else:
            print(f"Error: {args.input} is not an EDF file")
            return
    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith('.edf'):
                input_files.append(os.path.join(args.input, file))
        if not input_files:
            print(f"Error: No EDF files found in {args.input}")
            return
    else:
        print(f"Error: {args.input} does not exist")
        return
    
    print(f"Found {len(input_files)} EDF files to process")
    
    # Run inference
    print("\nRunning inference...")
    results = inference.predict_batch(input_files)
    
    # Generate report
    report_path = os.path.join(args.output, 'inference_report.json')
    inference.generate_report(results, report_path)
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        inference.create_visualization(results, args.output)
    
    print(f"\nInference complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()