# run_adversarial_attack.py
"""
Script to run adversarial attacks on audio source separation models using Monte Carlo optimization.
Uses the MonteCarloOptimizer class for flexible experimentation with different perturbation methods.
"""

import os
import subprocess
import argparse
from pathlib import Path
import numpy as np
import librosa
import mir_eval
import sys
# Import the optimizer


# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from optimizations.monte_carlo_optimizer import MonteCarloOptimizer, plot_results


# Global constraints parameters
EPSILON_INF = 0.002  # L-infinity constraint
TARGET_DB = -40      # L2 constraint in dB
WEIGHTED_LOSS = {    # Weights for combined loss
    'sdr': 0.6,
    'sir': 0.3, 
    'sar': 0.1
}

def initialize_globals(epsilon_inf=0.002, target_db=-40, weights=None):
    """Initialize global parameters for adversarial attack"""
    global EPSILON_INF, TARGET_DB, WEIGHTED_LOSS
    
    EPSILON_INF = epsilon_inf
    TARGET_DB = target_db
    
    if weights is not None:
        WEIGHTED_LOSS = weights
    
    print("Initialized global parameters:")
    print(f"  L-infinity constraint: {EPSILON_INF}")
    print(f"  L2 constraint (dB): {TARGET_DB}")
    print(f"  Loss weights: SDR={WEIGHTED_LOSS['sdr']}, SIR={WEIGHTED_LOSS['sir']}, SAR={WEIGHTED_LOSS['sar']}")

def demucs_model(input_file, output_dir):
    """
    Run Demucs source separation using subprocess to call the command-line interface
    
    Parameters:
    input_file (str): Path to input audio file
    output_dir (str): Directory to save separated stems
    
    Returns:
    bool: True if separation succeeded, False otherwise
    """
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the Demucs command
        cmd = ["demucs", "-n", "htdemucs_ft", "--out", output_dir, input_file]
        
        # Run the process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,  # Return strings rather than bytes
            shell=True  # Necessary on Windows
        )
        
        # Get output and error
        stdout, stderr = process.communicate()
        
        # Check if separation succeeded
        if process.returncode != 0:
            print(f"Demucs error: {stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error running Demucs subprocess: {e}")
        return False

def l2_norm(perturbation):
    """Calculate the L2 norm (Euclidean norm) of a perturbation"""
    return np.sqrt(np.sum(perturbation**2))

def l2_norm_db(perturbation, original_audio):
    """Calculate L2 norm of perturbation in decibels relative to original signal"""
    perturbation_norm = l2_norm(perturbation)
    original_norm = l2_norm(original_audio)
    
    if original_norm > 0 and perturbation_norm > 0:
        db = 20 * np.log10(perturbation_norm / original_norm)
    else:
        db = float('-inf')  # Handle zero case
        
    return db

def l2_constrain_perturbation(perturbation, original_audio, target_db=None):
    """Constrain a perturbation to a target dB level relative to original audio"""
    t_db = target_db if target_db is not None else TARGET_DB
    
    current_db = l2_norm_db(perturbation, original_audio)
    
    if current_db > t_db:  # Only scale down if too loud
        db_change = t_db - current_db
        scale_factor = 10 ** (db_change / 20)  # Convert dB to amplitude ratio
        perturbation = perturbation * scale_factor
    
    return perturbation

def estimate_quality(original_sources, estimated_sources):
    """
    Calculate weighted combination of separation metrics
    
    Parameters:
    original_sources: Source audio before separation
    estimated_sources: Source audio after separation
    
    Returns:
    numpy.ndarray: Weighted combination of SDR, SIR, and SAR for each source
    """
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(original_sources, estimated_sources)
    
    # Use global weights
    return (WEIGHTED_LOSS['sdr'] * sdr + 
            WEIGHTED_LOSS['sir'] * sir + 
            WEIGHTED_LOSS['sar'] * sar)

def generate_perturbation(original_audio):
    """Generate perturbation with both L-infinity and L2 constraints using global parameters"""
    # Create random perturbation
    perturbation = np.random.normal(0, 0.01, original_audio.shape)
    
    # Apply L-infinity constraint
    perturbation = np.clip(perturbation, -EPSILON_INF, EPSILON_INF)
    
    # Apply L2 constraint (energy)
    perturbation = l2_constrain_perturbation(perturbation, original_audio, TARGET_DB)
    
    # Double-check we still meet L-infinity after scaling
    perturbation = np.clip(perturbation, -EPSILON_INF, EPSILON_INF)
    
    # Calculate constraint metrics
    metrics = {
        'epsilon_inf': EPSILON_INF,
        'target_db': TARGET_DB,
        'l2_norm': l2_norm(perturbation),
        'linf_norm': np.max(np.abs(perturbation))
    }
    
    return perturbation, metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Adversarial Attack on Audio Source Separation')
    parser.add_argument('--input', required=True, help='Path to input audio file')
    parser.add_argument('--sources_dir', required=True, help='Directory containing original separated stems')
    parser.add_argument('--output_dir', default='./results', help='Directory to save results')
    parser.add_argument('--iterations', type=int, default=500, help='Number of Monte Carlo iterations')
    parser.add_argument('--epsilon_inf', type=float, default=0.002, help='L-infinity constraint (relative)')
    parser.add_argument('--target_db', type=float, default=-40, help='L2 constraint (dB)')
    parser.add_argument('--stop_criteria', type=float, default=4.0, help='Early stopping criteria')
    parser.add_argument('--amplify', action='store_true', help='Create amplified version of perturbation')
    parser.add_argument('--weights', nargs=3, type=float, metavar=('SDR', 'SIR', 'SAR'),
                       default=[0.6, 0.3, 0.1], help='Weights for SDR, SIR, and SAR in combined metric')
    parser.add_argument('--save_all', action='store_true', help='Save all iterations (not just the best)')
    args = parser.parse_args()
    
    # Initialize globals with command line arguments
    initialize_globals(
        epsilon_inf=args.epsilon_inf,
        target_db=args.target_db,
        weights={'sdr': args.weights[0], 'sir': args.weights[1], 'sar': args.weights[2]}
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Set up paths for original sources
    expected_sources = ["drums", "bass", "vocals", "other"]
    sources_paths = []
    
    for source_name in expected_sources:
        source_path = os.path.join(args.sources_dir, f"{source_name}.wav")
        if os.path.exists(source_path):
            sources_paths.append(source_path)
        else:
            print(f"Warning: Expected source file '{source_name}.wav' not found in {args.sources_dir}")

    if not sources_paths:
        raise ValueError(f"No source files found in {args.sources_dir}")
    
    print(f"Found {len(sources_paths)} source files: {[Path(p).name for p in sources_paths]}")
    print(f"Starting Monte Carlo optimization with {args.iterations} iterations")
    
    # Create the optimizer instance
    optimizer = MonteCarloOptimizer(
        model_fn=demucs_model,
        estimation_fn=estimate_quality,
        perturbation_fn=generate_perturbation
    )
    # Run Monte Carlo optimization
    results, best_perturbation, best_score, best_audio, csv_results = optimizer.optimize(
    args.input, 
    sources_paths, 
    args.output_dir,
    iterations=args.iterations,
    stop_criteria=args.stop_criteria,
    save_all_iterations=args.save_all
    )

    # Load original audio for plotting
    original_audio, sr = librosa.load(args.input, sr=None)

    # Plot results with csv_results included
    plot_results(results, original_audio, best_audio, sr, args.output_dir, csv_results)
    
  
    # Print summary
    print("\n--- Attack Results ---")
    print(f"Best combined score achieved: {best_score:.4f}")
    print(f"Original audio: {args.input}")
    print(f"Best adversarial audio saved to: {os.path.join(args.output_dir, 'best_adversarial_audio.wav')}")
    print(f"Best perturbation saved to: {os.path.join(args.output_dir, 'best_perturbation.wav')}")
    print(f"Plots saved to: {os.path.join(args.output_dir, 'plots')}")
    
    
    # Try to calculate perceptual metrics if available
    try:
        from pystoi import stoi
        
        # Calculate STOI (works with any sample rate)
        stoi_score = stoi(original_audio, best_audio, sr, extended=False)
        print(f"STOI score: {stoi_score:.4f}")
        
        # For PESQ, we would need to resample to 16kHz first
        try:
            from pesq import pesq
            print("Calculating PESQ (resampling to 16kHz)...")
            
            # Resample for PESQ calculation
            orig_16k = librosa.resample(original_audio, orig_sr=sr, target_sr=16000)
            adv_16k = librosa.resample(best_audio, orig_sr=sr, target_sr=16000)
            
            # Calculate PESQ
            pesq_score = pesq(16000, orig_16k, adv_16k, 'wb')
            print(f"PESQ score: {pesq_score:.4f}")
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
            
    except ImportError:
        print("Perceptual metrics not calculated (pystoi/pesq not installed)")

if __name__ == "__main__":
    main()