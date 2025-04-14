# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:01:11 2025

@author: igor
"""

# monte_carlo_optimizer.py
"""
Monte Carlo optimization framework for adversarial attacks on audio source separation.

This module provides a flexible framework for generating adversarial perturbations
using Monte Carlo sampling methods to attack audio source separation models.
"""

import os
import shutil
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Callable, Any, Optional, Union

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt


class MonteCarloOptimizer:
    """Monte Carlo optimization framework for generating adversarial audio examples."""
    
    def __init__(
        self,
        model_fn: Callable,
        estimation_fn: Callable,
        perturbation_fn: Callable,
        source_names: List[str] = ["drums", "bass", "vocals", "other"]
    ):
        """
        Initialize the Monte Carlo optimizer.
        
        Args:
            model_fn: Function that runs the separation model
            estimation_fn: Function that evaluates separation quality
            perturbation_fn: Function that generates perturbations
            source_names: Names of the source stems to separate
        """
        self.model_fn = model_fn
        self.estimation_fn = estimation_fn
        self.perturbation_fn = perturbation_fn
        self.source_names = source_names
    
    def optimize(
        self,
        original_audio_path: str,
        original_sources_paths: List[str],
        output_dir: str,
        iterations: int = 500,
        stop_criteria: float = 4.0,
        save_all_iterations: bool = False,
        verbose: bool = True
    ) -> Tuple[List, np.ndarray, float, np.ndarray]:
        """
        Run Monte Carlo optimization to find adversarial perturbations.
        
        Args:
            original_audio_path: Path to the original audio file
            original_sources_paths: List of paths to original separated sources
            output_dir: Directory to save results
            iterations: Number of Monte Carlo iterations to run
            stop_criteria: Early stopping threshold for metrics
            save_all_iterations: Whether to save all iterations' results
            verbose: Whether to print progress updates
            
        Returns:
            Tuple containing:
            - List of score results for each iteration
            - Best perturbation found
            - Best score achieved
            - Best adversarial audio
        """
        # Load original audio
        original_audio, sr = librosa.load(original_audio_path, sr=None)
        original_audio = np.array(original_audio)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original audio for reference
        original_path = os.path.join(output_dir, "original_audio.wav")
        sf.write(original_path, original_audio, sr)
        
        # Load original sources for comparison
        original_sources = []
        for path in original_sources_paths:
            source_audio, _ = librosa.load(path, sr=sr)  # Use same sr as original
            original_sources.append(source_audio)
        
        original_sources = np.array(original_sources)
        
        # Setup tracking variables
        best_perturbation = None
        best_score = np.inf
        best_audio = None
        best_iter = -1
        results = []
        csv_results = []
        
        # Create temp directory for intermediate results
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Run iterations
        for i in range(iterations):
            # Generate perturbation
            if(i%10 == 0 and i>0):
                print(f'iteration: {i}')
                print(f'best_score: {best_score}')
            perturbation, perturbation_metrics = self.perturbation_fn(original_audio)
            
            # Apply perturbation
            perturbed_audio = original_audio + perturbation
            
            # Save perturbed audio to temp file
            temp_file = os.path.join(temp_dir, "perturbed_input.wav")
            sf.write(temp_file, perturbed_audio, sr)
            
            # Run model on perturbed audio
            temp_output_dir = os.path.join(temp_dir, f"iter_{i}")
            os.makedirs(temp_output_dir, exist_ok=True)
            success = self.model_fn(temp_file, temp_output_dir)
            
            if not success:
                if verbose:
                    print(f"Separation failed for iteration {i}")
                continue
            
            # Load estimated sources
            track_name = os.path.splitext(os.path.basename(temp_file))[0]
            track_dir = os.path.join(temp_output_dir, "htdemucs_ft", track_name)
            estimated_sources = []
            
            # Ensure same order as original sources
            for source_name in self.source_names:
                source_path = os.path.join(track_dir, f"{source_name}.wav")
                
                if not os.path.exists(source_path):
                    if verbose:
                        print(f"Warning: Source file not found: {source_path}")
                    continue
                    
                source_audio, _ = librosa.load(source_path, sr=sr)
                estimated_sources.append(source_audio)
            
            # Skip if no sources were found
            if len(estimated_sources) == 0:
                if verbose:
                    print(f"No source files found for iteration {i}")
                continue
                
            estimated_sources = np.array(estimated_sources)
            
            # Check if shapes match before calculating metrics
            if original_sources.shape[0] != estimated_sources.shape[0]:
                if verbose:
                    print(f"Warning: Number of sources doesn't match. Original: {original_sources.shape[0]}, Estimated: {estimated_sources.shape[0]}")
                continue
                
            # Ensure length matches by trimming
            min_length = min(original_sources.shape[1], estimated_sources.shape[1])
            orig_trimmed = original_sources[:, :min_length]
            est_trimmed = estimated_sources[:, :min_length]
            
            # Calculate score
            score = self.estimation_fn(orig_trimmed, est_trimmed)
            
            # Calculate mean score
            mean_score = np.mean(score)
            
            # Record for CSV
            iteration_result = {
                'iteration': i,
                'mean_score': mean_score,
                'best_score_so_far': best_score,
                'sdr_per_source': score
            }
            iteration_result.update(perturbation_metrics)
            csv_results.append(iteration_result)
            
            # Update best result
            if mean_score < best_score:
                best_score = mean_score
                best_audio = perturbed_audio
                best_perturbation = perturbation
                best_iter = i
            
            results.append(score)
            
            if verbose and i % 10 == 0 and i > 0:
                print(f"Iteration {i}: best result: {best_score:.4f}")
                
            # Early stopping
            if best_score < stop_criteria:
                if verbose:
                    print(f"Reached stop criteria at iteration {i}")
                break
        
        # Save results
        self._save_best_results(
            best_perturbation, 
            best_audio, 
            best_iter,
            temp_dir, 
            output_dir, 
            sr
        )
        
        # Save CSV results
        self._save_results_to_csv(csv_results, output_dir)

        
        # Cleanup temp directory to save space
        if not save_all_iterations:
            shutil.rmtree(temp_dir, ignore_errors=True)
                
        self._save_results_to_csv(csv_results, output_dir)

        # Return the results for further analysis
        return results, best_perturbation, best_score, best_audio, csv_results  # Added csv_results    
    
    def _save_best_results(
        self, 
        best_perturbation: np.ndarray, 
        best_audio: np.ndarray, 
        best_iter: int,
        temp_dir: str, 
        output_dir: str, 
        sr: int
    ) -> None:
        """Save the best results from the optimization."""
        if best_perturbation is None:
            return
            
        # 1. Save the best perturbation
        best_pert_path = os.path.join(output_dir, "best_perturbation.wav")
        sf.write(best_pert_path, best_perturbation, sr)
        
        # 2. Save the original audio with the best perturbation applied
        best_audio_path = os.path.join(output_dir, "best_adversarial_audio.wav")
        sf.write(best_audio_path, best_audio, sr)
        
        # 3. Copy the separated sources from the best iteration
        best_sources_dir = os.path.join(output_dir, "best_separated_sources")
        os.makedirs(best_sources_dir, exist_ok=True)
        
        temp_track_dir = os.path.join(temp_dir, f"iter_{best_iter}", "htdemucs_ft", "perturbed_input")
        
        if os.path.exists(temp_track_dir):
            for source_name in self.source_names:
                source_file = os.path.join(temp_track_dir, f"{source_name}.wav")
                if os.path.exists(source_file):
                    dest_file = os.path.join(best_sources_dir, f"{source_name}.wav")
                    shutil.copy2(source_file, dest_file)
        else:
            print(f"Warning: Could not find separated sources for best iteration at {temp_track_dir}")
    
    def _save_results_to_csv(self, results_data: List[Dict], output_dir: str) -> Optional[str]:
        """Save Monte Carlo optimization results to CSV."""
        csv_path = os.path.join(output_dir, "monte_carlo_results.csv")
        
        if len(results_data) == 0:
            return None
            
        # Get all column names from the first result
        headers = list(results_data[0].keys())
        # Remove 'score_per_source' from headers as it will be expanded
        if 'score_per_source' in headers:
            headers.remove('score_per_source')
        elif 'sdr_per_source' in headers:  # Handle legacy naming
            headers.remove('sdr_per_source')
        
        # Add source-specific columns using actual source names
        for source_name in self.source_names:
            headers.append(source_name)
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for row in results_data:
                row_dict = {k: v for k, v in row.items() if k not in ('score_per_source', 'sdr_per_source')}
                
                # Add per-source values if available using actual source names
                score_values = None
                if 'score_per_source' in row and isinstance(row['score_per_source'], np.ndarray):
                    score_values = row['score_per_source']
                elif 'sdr_per_source' in row and isinstance(row['sdr_per_source'], np.ndarray):
                    score_values = row['sdr_per_source']
                    
                if score_values is not None:
                    for i, val in enumerate(score_values):
                        if i < len(self.source_names):
                            row_dict[self.source_names[i]] = val
                
                writer.writerow(row_dict)
        
        print(f"Results saved to {csv_path}")
        return csv_path


def plot_results(
    results: List, 
    original_audio: np.ndarray, 
    perturbed_audio: np.ndarray, 
    sr: int, 
    output_dir: str,
    csv_results: Optional[List[Dict]] = None  # Added parameter for best score tracking

) -> None:
    """
    Plot and save results from the Monte Carlo optimization.
    """
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate perturbation
    perturbation = perturbed_audio - original_audio
    
    
    # 1. Plot all scores progression over iterations
    plt.figure(figsize=(10, 6))
    if isinstance(results[0], np.ndarray):
        # If results contain multiple sources, calculate mean
        mean_results = [np.mean(res) for res in results]
        for i in range(len(results[0])):
            source_results = [res[i] for res in results]
            source_name = f"Source {i+1}"
            if i < len(["drums", "bass", "vocals", "other"]):
                source_name = ["drums", "bass", "vocals", "other"][i]
            plt.plot(source_results, alpha=0.5, label=source_name)
        plt.plot(mean_results, 'k-', linewidth=2, label='Mean')
    else:
        # If results are already mean values
        plt.plot(results, 'b-', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Score (lower is better)')
    plt.title('All Scores by Iteration (Lower is Better)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'all_scores_progression.png'), dpi=300)
    
    # 1b. Plot best score progression over iterations
    if csv_results:
        plt.figure(figsize=(10, 6))
        best_scores = [row.get('best_score_so_far', float('inf')) for row in csv_results]
        iterations = [row.get('iteration', i) for i, row in enumerate(csv_results)]
        
        plt.plot(iterations, best_scores, 'r-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Score So Far (lower is better)')
        plt.title('Best Score Progression (Lower is Better)')
        plt.grid(True, alpha=0.3)
        
        # Add annotations for significant improvements
        significant_improvements = []
        prev_best = float('inf')
        for i, score in enumerate(best_scores):
            if i > 0 and score < prev_best and (prev_best - score) > 0.05:
                significant_improvements.append((iterations[i], score))
                prev_best = score
        
        # Limit annotations to avoid clutter (max 10)
        if len(significant_improvements) > 10:
            improvement_values = [imp[1] for imp in significant_improvements]
            threshold = sorted(improvement_values)[len(significant_improvements) - 10]
            significant_improvements = [imp for imp in significant_improvements if imp[1] <= threshold]
        
        for iter_num, score in significant_improvements:
            plt.annotate(f'{score:.2f}', 
                        xy=(iter_num, score),
                        xytext=(5, -5),
                        textcoords='offset points',
                        fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'best_score_progression.png'), dpi=300)
    
    
    # 1. Plot metrics progression over iterations
    plt.figure(figsize=(10, 6))
    if isinstance(results[0], np.ndarray):
        # If results contain multiple sources, calculate mean
        mean_results = [np.mean(res) for res in results]
        for i in range(len(results[0])):
            source_results = [res[i] for res in results]
            plt.plot(source_results, alpha=0.5, label=f'Source {i+1}')
        plt.plot(mean_results, 'k-', linewidth=2, label='Mean')
    else:
        # If results are already mean values
        plt.plot(results, 'b-', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Combined Score (lower is better)')
    plt.title('Optimization Progress (Lower is Better for Adversarial Attack)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'optimization_progress.png'), dpi=300)
    
    # 2. Plot waveform with perturbation overlay
    # For visualization clarity, we'll plot a subset if audio is very long
    MAX_SAMPLES = 100000  # ~2.3 seconds at 44.1kHz
    
    if len(original_audio) > MAX_SAMPLES:
        # Find a section with significant perturbation
        perturbation_energy = np.abs(perturbation) ** 2
        window_size = MAX_SAMPLES
        
        # Find highest energy section
        energy_windows = []
        for i in range(0, len(perturbation) - window_size, window_size // 2):
            energy_windows.append((i, np.sum(perturbation_energy[i:i+window_size])))
        
        start_idx = max(energy_windows, key=lambda x: x[1])[0]
        end_idx = min(start_idx + window_size, len(original_audio))
        
        # Extract section
        plot_original = original_audio[start_idx:end_idx]
        plot_perturbation = perturbation[start_idx:end_idx]
        time_axis = np.arange(start_idx, end_idx) / sr
    else:
        plot_original = original_audio
        plot_perturbation = perturbation
        time_axis = np.arange(len(original_audio)) / sr
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, plot_original, 'b-', alpha=0.7, label='Original Audio')
    # Amplify perturbation by 50x to make it visible
    plt.plot(time_axis, plot_perturbation * 50, 'r-', alpha=0.7, label='Perturbation (×50)')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Original Waveform with Perturbation Overlay')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'waveform_comparison.png'), dpi=300)
    
    # 3. Plot spectrograms for comparison
    plt.figure(figsize=(15, 10))
    
    # Original audio spectrogram
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Audio Spectrogram')
    
    # Perturbed audio spectrogram
    plt.subplot(2, 1, 2)
    D_perturbed = librosa.amplitude_to_db(np.abs(librosa.stft(perturbed_audio)), ref=np.max)
    librosa.display.specshow(D_perturbed, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Adversarial Audio Spectrogram')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'spectrogram_comparison.png'), dpi=300)
    
    # 4. Plot perturbation itself
    plt.figure(figsize=(10, 6))
    
    # Time domain
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(perturbation)) / sr, perturbation)
    plt.title('Perturbation in Time Domain')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Frequency domain
    plt.subplot(2, 1, 2)
    D_pert = librosa.amplitude_to_db(np.abs(librosa.stft(perturbation)), ref=np.max)
    librosa.display.specshow(D_pert, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Perturbation Spectrogram')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'perturbation_analysis.png'), dpi=300)
    
    plt.close('all')