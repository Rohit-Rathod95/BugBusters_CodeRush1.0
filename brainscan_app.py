"""
BrainScan - GUI Application for EEG Classification
A user-friendly interface for EEG analysis and classification.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime
import json
import pickle
import torch

# Mock EEGInference class for demonstration (replace with your actual implementation)
class EEGInference:
    """
    Mock inference class - replace with your actual implementation
    """
    def __init__(self, model_path, cohort_stats_path=None):
        self.model_path = model_path
        self.cohort_stats_path = cohort_stats_path
        self.cohort_stats = None
        
        # Load cohort stats if provided
        if cohort_stats_path:
            self.load_cohort_stats()
    
    def load_cohort_stats(self):
        """Load cohort statistics from file"""
        try:
            if self.cohort_stats_path.endswith('.pkl'):
                with open(self.cohort_stats_path, 'rb') as f:
                    self.cohort_stats = pickle.load(f)
            elif self.cohort_stats_path.endswith('.json'):
                with open(self.cohort_stats_path, 'r') as f:
                    self.cohort_stats = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cohort stats: {e}")
            self.cohort_stats = None
    
    def predict_single_file(self, file_path):
        """
        Mock prediction method - replace with your actual implementation
        """
        import time
        import random
        
        # Simulate processing time
        time.sleep(2)
        
        # Mock results - replace with actual inference logic
        n_epochs = random.randint(50, 200)
        prob_affected = random.random()
        
        # Mock individual predictions
        individual_preds = [random.random() for _ in range(n_epochs)]
        
        # Mock severity based on probability
        if prob_affected < 0.3:
            severity = "Normal"
        elif prob_affected < 0.5:
            severity = "Mild"
        elif prob_affected < 0.7:
            severity = "Moderate"
        else:
            severity = "Severe"
        
        # Mock metadata features
        metadata_features = {
            'delta_power': random.uniform(0.1, 0.5),
            'theta_power': random.uniform(0.05, 0.3),
            'alpha_power': random.uniform(0.1, 0.4),
            'beta_power': random.uniform(0.1, 0.3),
            'gamma_power': random.uniform(0.05, 0.2)
        }
        
        return {
            'probability_affected': prob_affected,
            'severity': severity,
            'n_epochs': n_epochs,
            'individual_predictions': individual_preds,
            'metadata_features': metadata_features
        }


class BrainScanApp:
    """
    Main GUI Application for EEG Classification.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("BrainScan - EEG Classification System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Application state
        self.inference_engine = None
        self.current_results = None
        self.model_path = None
        self.cohort_stats_path = None
        
        # Setup GUI
        self.setup_styles()
        self.create_menu()
        self.create_main_interface()
        
    def setup_styles(self):
        """Configure GUI styles."""
        self.style = ttk.Style()
        
        # Try to use a modern theme, fall back to default if not available
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Status.TLabel', foreground='blue')
        self.style.configure('Error.TLabel', foreground='red')
        self.style.configure('Success.TLabel', foreground='green')
        self.style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    
    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Load Cohort Stats", command=self.load_cohort_stats)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
    
    def create_main_interface(self):
        """Create the main application interface."""
        
        # Create main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="BrainScan EEG Classification", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Control Panel (Left side)
        self.create_control_panel(main_frame)
        
        # Results Panel (Right side)
        self.create_results_panel(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_control_panel(self, parent):
        """Create the control panel for user inputs."""
        
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model Selection
        ttk.Label(control_frame, text="Model Configuration", style='Heading.TLabel').grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        ttk.Label(control_frame, text="Model File:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.model_path_var = tk.StringVar(value="No model loaded")
        ttk.Label(control_frame, textvariable=self.model_path_var, 
                 foreground='gray').grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Button(control_frame, text="Browse Model", 
                  command=self.load_model).grid(row=2, column=0, columnspan=2, 
                                               sticky=tk.W, pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # File Input
        ttk.Label(control_frame, text="EEG File Analysis", style='Heading.TLabel').grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        ttk.Label(control_frame, text="Select EDF File:").grid(row=5, column=0, sticky=tk.W, pady=2)
        
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(0, weight=1)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=40)
        self.file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_edf_file).grid(row=0, column=1)
        
        # Analysis Button
        self.analyze_button = ttk.Button(control_frame, text="Analyze EEG", 
                                        command=self.run_analysis, 
                                        style='Accent.TButton')
        self.analyze_button.grid(row=7, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        self.analyze_button.configure(state='disabled')
        
        # Progress Bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    def create_results_panel(self, parent):
        """Create the results display panel."""
        
        results_frame = ttk.LabelFrame(parent, text="Analysis Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results Summary
        summary_frame = ttk.Frame(results_frame)
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        summary_frame.columnconfigure(1, weight=1)
        
        self.create_results_summary(summary_frame)
        
        # Visualization Area
        viz_frame = ttk.LabelFrame(results_frame, text="Visualization")
        viz_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        self.create_visualization_area(viz_frame)
    
    def create_results_summary(self, parent):
        """Create the results summary section."""
        
        # Prediction Result
        ttk.Label(parent, text="Prediction:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.prediction_var = tk.StringVar(value="No analysis performed")
        self.prediction_label = ttk.Label(parent, textvariable=self.prediction_var, 
                                         font=('Arial', 12, 'bold'))
        self.prediction_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Probability
        ttk.Label(parent, text="Probability:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.probability_var = tk.StringVar(value="--")
        ttk.Label(parent, textvariable=self.probability_var).grid(row=1, column=1, 
                                                                 sticky=tk.W, padx=(10, 0))
        
        # Severity
        ttk.Label(parent, text="Severity:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.severity_var = tk.StringVar(value="--")
        self.severity_label = ttk.Label(parent, textvariable=self.severity_var)
        self.severity_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Epochs Analyzed
        ttk.Label(parent, text="Epochs:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value="--")
        ttk.Label(parent, textvariable=self.epochs_var).grid(row=3, column=1, 
                                                            sticky=tk.W, padx=(10, 0))
    
    def create_visualization_area(self, parent):
        """Create the matplotlib visualization area."""
        
        # Create matplotlib figure
        plt.style.use('default')  # Ensure we're using a compatible style
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # Initialize empty plots
        self.ax1.text(0.5, 0.5, 'No data to display', ha='center', va='center', 
                     transform=self.ax1.transAxes, fontsize=12, color='gray')
        self.ax1.set_title('Bandpower Comparison')
        
        self.ax2.text(0.5, 0.5, 'No data to display', ha='center', va='center', 
                     transform=self.ax2.transAxes, fontsize=12, color='gray')
        self.ax2.set_title('Prediction Confidence')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add navigation toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
    def create_status_bar(self, parent):
        """Create the status bar."""
        
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready - Load a model to begin")
        ttk.Label(status_frame, textvariable=self.status_var, 
                 style='Status.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        # Add timestamp
        self.timestamp_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.timestamp_var, 
                 font=('Arial', 8)).grid(row=0, column=1, sticky=tk.E)
    
    def load_model(self):
        """Load a trained model."""
        
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            initialdir="."
        )
        
        if file_path:
            try:
                self.inference_engine = EEGInference(file_path, self.cohort_stats_path)
                self.model_path = file_path
                self.model_path_var.set(os.path.basename(file_path))
                self.status_var.set("Model loaded successfully")
                self.analyze_button.configure(state='normal')
                
                # Update timestamp
                self.timestamp_var.set(f"Model loaded: {datetime.now().strftime('%H:%M:%S')}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.status_var.set("Error loading model")
    
    def load_cohort_stats(self):
        """Load cohort statistics."""
        
        file_path = filedialog.askopenfilename(
            title="Select Cohort Statistics File",
            filetypes=[("Pickle Files", "*.pkl"), ("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir="."
        )
        
        if file_path:
            self.cohort_stats_path = file_path
            try:
                # Reload inference engine with cohort stats if model already loaded
                if self.inference_engine:
                    self.inference_engine = EEGInference(self.model_path, self.cohort_stats_path)
                self.status_var.set("Cohort statistics loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load cohort statistics:\n{str(e)}")
                self.status_var.set("Error loading cohort statistics")
    
    def browse_edf_file(self):
        """Browse for EDF file."""
        
        file_path = filedialog.askopenfilename(
            title="Select EDF File",
            filetypes=[("EDF Files", "*.edf"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
    
    def run_analysis(self):
        """Run EEG analysis in a separate thread."""
        
        if not self.inference_engine:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an EDF file")
            return
        
        # Disable button and start progress
        self.analyze_button.configure(state='disabled')
        self.progress.start()
        self.status_var.set("Analyzing EEG data...")
        
        # Run analysis in separate thread
        analysis_thread = threading.Thread(target=self._perform_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _perform_analysis(self):
        """Perform the actual analysis (runs in separate thread)."""
        
        try:
            # Run inference
            results = self.inference_engine.predict_single_file(self.file_path_var.get())
            
            # Update UI in main thread
            self.root.after(0, self._update_results, results)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, self._show_error, error_msg)
    
    def _update_results(self, results):
        """Update the UI with analysis results."""
        
        self.progress.stop()
        self.analyze_button.configure(state='normal')
        
        if not isinstance(results, dict):
            self._show_error("Unexpected result format from inference engine.")
            return
        
        if 'error' in results:
            self._show_error(results['error'])
            return
        
        # Defensive defaults
        prob = float(results.get('probability_affected', 0.0))
        severity = str(results.get('severity', '--'))
        n_epochs = int(results.get('n_epochs', 0))
        
        self.current_results = results
        
        # Update summary
        prediction = "AFFECTED" if prob > 0.5 else "HEALTHY"
        self.prediction_var.set(prediction)
        
        # Set prediction label color
        color = 'red' if prediction == "AFFECTED" else 'green'
        self.prediction_label.configure(foreground=color)
        
        self.probability_var.set(f"{prob:.3f}")
        self.severity_var.set(severity)
        self.epochs_var.set(str(n_epochs))
        
        # Set severity color
        severity_colors = {
            'Normal': 'green',
            'Mild': 'orange',
            'Moderate': 'darkorange',
            'Severe': 'red',
            'Critical': 'darkred'
        }
        self.severity_label.configure(foreground=severity_colors.get(severity, 'black'))
        
        # Update visualizations
        self._update_visualizations(results)
        
        self.status_var.set("Analysis completed successfully")
        self.timestamp_var.set(f"Completed: {datetime.now().strftime('%H:%M:%S')}")
    
    def _update_visualizations(self, results):
        """Update the matplotlib visualizations."""
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Bandpower comparison (if cohort stats available)
        if hasattr(self.inference_engine, "cohort_stats") and self.inference_engine.cohort_stats and 'metadata_features' in results:
            try:
                self._plot_bandpower_comparison(results)
            except Exception as e:
                self.ax1.text(0.5, 0.5, f'Error plotting bandpower:\n{e}', 
                              ha='center', va='center', transform=self.ax1.transAxes)
                self.ax1.set_title('Bandpower Comparison')
        else:
            self.ax1.text(0.5, 0.5, 'Cohort statistics not available\nfor comparison', 
                         ha='center', va='center', transform=self.ax1.transAxes)
            self.ax1.set_title('Bandpower Comparison')
        
        # Plot 2: Prediction confidence over epochs
        if 'individual_predictions' in results and isinstance(results['individual_predictions'], (list, np.ndarray)):
            try:
                self._plot_prediction_confidence(results)
            except Exception as e:
                self.ax2.text(0.5, 0.5, f'Error plotting confidence:\n{e}', 
                              ha='center', va='center', transform=self.ax2.transAxes)
                self.ax2.set_title('Prediction Confidence')
        else:
            self.ax2.text(0.5, 0.5, 'Individual predictions\nnot available', 
                         ha='center', va='center', transform=self.ax2.transAxes)
            self.ax2.set_title('Prediction Confidence')
        
        # Refresh canvas
        self.canvas.draw()
    
    def _plot_bandpower_comparison(self, results):
        """Plot bandpower comparison with cohort statistics."""
        
        metadata = results.get('metadata_features', {})
        cohort_stats = self.inference_engine.cohort_stats or {}
        
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        power_keys = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
        
        # Gather values with safe defaults
        user_powers = [float(metadata.get(key, np.nan)) for key in power_keys]
        
        def get_group(stats_dict, group_name, agg):
            return [float(stats_dict.get(group_name, {}).get(agg, {}).get(k, np.nan)) for k in power_keys]
        
        healthy_means = get_group(cohort_stats, 'healthy', 'mean')
        affected_means = get_group(cohort_stats, 'affected', 'mean')
        healthy_stds = get_group(cohort_stats, 'healthy', 'std')
        affected_stds = get_group(cohort_stats, 'affected', 'std')
        
        x = np.arange(len(bands))
        width = 0.25
        
        # Bars: user (no error bars), healthy mean (±std), affected mean (±std)
        self.ax1.bar(x - width, user_powers, width, label='This file')
        if not np.all(np.isnan(healthy_means)):
            if not np.all(np.isnan(healthy_stds)):
                self.ax1.bar(x, healthy_means, width, yerr=healthy_stds, capsize=4, label='Healthy (mean±std)')
            else:
                self.ax1.bar(x, healthy_means, width, label='Healthy (mean)')
        if not np.all(np.isnan(affected_means)):
            if not np.all(np.isnan(affected_stds)):
                self.ax1.bar(x + width, affected_means, width, yerr=affected_stds, capsize=4, label='Affected (mean±std)')
            else:
                self.ax1.bar(x + width, affected_means, width, label='Affected (mean)')
        
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(bands)
        self.ax1.set_ylabel('Relative Band Power')
        self.ax1.set_title('Bandpower Comparison')
        self.ax1.legend(loc='best')
        self.ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    def _plot_prediction_confidence(self, results):
        """Plot prediction confidence (affected probability) across epochs."""
        preds = np.array(results.get('individual_predictions', []), dtype=float)
        if preds.size == 0:
            self.ax2.text(0.5, 0.5, 'No per-epoch predictions available', 
                          ha='center', va='center', transform=self.ax2.transAxes)
            self.ax2.set_title('Prediction Confidence')
            return
        
        epochs = np.arange(1, len(preds) + 1)
        self.ax2.plot(epochs, preds, marker='o', markersize=3)
        
        # Rolling mean for smoothing (window=5 if enough points)
        if len(preds) >= 5:
            roll = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().to_numpy()
            self.ax2.plot(epochs, roll, linestyle='--', label='Rolling mean (w=5)')
            self.ax2.legend(loc='best')
        
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('P(Affected)')
        self.ax2.set_ylim(0.0, 1.0)
        self.ax2.grid(True, linestyle='--', alpha=0.3)
        self.ax2.set_title('Prediction Confidence')
    
    def export_results(self):
        """Export current results to disk (JSON summary, CSV per-epoch preds, PNG of plots)."""
        if not self.current_results:
            messagebox.showwarning("No Results", "Run an analysis before exporting.")
            return
        
        # Choose export directory
        export_dir = filedialog.askdirectory(title="Select Export Folder")
        if not export_dir:
            return
        
        # Base name from EDF file or timestamp
        base = os.path.splitext(os.path.basename(self.file_path_var.get() or "session"))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{base}_{timestamp}"
        
        # 1) JSON summary
        summary = {
            "file": self.file_path_var.get(),
            "model_path": self.model_path,
            "cohort_stats_path": self.cohort_stats_path,
            "completed_at": datetime.now().isoformat(timespec='seconds'),
            "prediction": "AFFECTED" if float(self.current_results.get("probability_affected", 0)) > 0.5 else "HEALTHY",
            "probability_affected": float(self.current_results.get("probability_affected", 0)),
            "severity": self.current_results.get("severity", None),
            "n_epochs": int(self.current_results.get("n_epochs", 0)),
            "metadata_features": self.current_results.get("metadata_features", {}),
        }
        json_path = os.path.join(export_dir, f"{stem}_summary.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self._show_error(f"Failed to write summary JSON: {e}")
            return
        
        # 2) CSV per-epoch predictions (if available)
        csv_path = None
        if isinstance(self.current_results.get("individual_predictions"), (list, np.ndarray)):
            preds = list(map(float, self.current_results["individual_predictions"]))
            df = pd.DataFrame({"epoch": np.arange(1, len(preds)+1), "prob_affected": preds})
            csv_path = os.path.join(export_dir, f"{stem}_per_epoch.csv")
            try:
                df.to_csv(csv_path, index=False)
            except Exception as e:
                self._show_error(f"Failed to write per-epoch CSV: {e}")
                return
        
        # 3) Save current figure as PNG
        png_path = os.path.join(export_dir, f"{stem}_plots.png")
        try:
            # Temporarily draw on a non-interactive canvas for saving
            self.fig.tight_layout()
            self.fig.savefig(png_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            self._show_error(f"Failed to save plots image: {e}")
            return
        
        msg = f"Exported:\n- {os.path.basename(json_path)}"
        if csv_path:
            msg += f"\n- {os.path.basename(csv_path)}"
        msg += f"\n- {os.path.basename(png_path)}"
        messagebox.showinfo("Export Complete", msg)
        self.status_var.set("Results exported successfully")
    
    def show_about(self):
        """Show About dialog."""
        messagebox.showinfo(
            "About BrainScan",
            "BrainScan - EEG Classification System\n"
            "GUI for EEG analysis and classification.\n\n"
            "© 2025 BrainScan Team"
        )
    
    def show_user_guide(self):
        """Show a brief user guide."""
        guide = (
            "User Guide (Quick Start)\n\n"
            "1) File → Load Model (.pth)\n"
            "2) (Optional) File → Load Cohort Stats (.pkl/.json) for bandpower comparison\n"
            "3) Select EDF file using the Browse button\n"
            "4) Click 'Analyze EEG' and wait for completion\n"
            "5) View Prediction and plots on the right\n"
            "6) File → Export Results to save JSON summary, per-epoch CSV, and plot image\n\n"
            "Notes:\n"
            "- Probability shows P(Affected). Prediction is AFFECTED if > 0.5, else HEALTHY.\n"
            "- Cohort stats structure expected: {healthy:{mean:{...},std:{...}}, affected:{mean:{...},std:{...}}}.\n"
        )
        messagebox.showinfo("User Guide", guide)
    
    def _show_error(self, msg):
        """Show an error and update status line."""
        messagebox.showerror("Error", msg)
        self.status_var.set("Error: " + msg)
        self.timestamp_var.set(f"{datetime.now().strftime('%H:%M:%S')}")
        self.progress.stop()
        self.analyze_button.configure(state='normal')


def main():
    """Main entry point for the application."""
    try:
        root = tk.Tk()
        app = BrainScanApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()