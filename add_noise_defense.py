# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 10:58:01 2025

@author: igor
"""

#!/usr/bin/env python3
"""
Batch noise processor for song folders.

This script processes a folder structure where each subfolder contains
a file named 'perturbed_mixture.wav'. It adds L-infinity bounded noise
to each file and saves the result in an output folder, with each file
named after the subfolder it was in.

Usage:
    python batch_noise_processor.py --input-dir songs_folder --output-dir output_folder --epsilon 0.01
"""

import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
import pandas as pd

def add_noise_with_linf_constraint(audio, epsilon):
    """
    Add noise to audio with L-infinity constraint.
    
    Args:
        audio: Audio signal as numpy array (can be mono or stereo)
        epsilon: Maximum absolute value of noise (L-infinity bound)
        
    Returns:
        Perturbed audio with added noise
    """
    # Generate random noise of the same shape as the audio
    noise = np.random.uniform(-1, 1, size=audio.shape)
    
    # Scale noise to respect the L-infinity constraint
    # First normalize to [-1, 1]
    max_abs_val = np.max(np.abs(noise))
    if max_abs_val > 0:  # Avoid division by zero
        noise = noise / max_abs_val
    
    # Then scale by epsilon to respect the L-infinity bound
    noise = noise * epsilon
    
    # Add the noise to the audio
    perturbed_audio = audio + noise
    
    # Clip to [-1, 1] to avoid distortion (assuming audio is normalized)
    perturbed_audio = np.clip(perturbed_audio, -1.0, 1.0)
    
    return perturbed_audio, noise

def calculate_snr(original, noise):
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def process_song_folders(input_dir, output_dir, epsilon):
    """
    Process all song folders, each containing a perturbed_mixture.wav file.
    
    Args:
        input_dir: Directory containing song folders
        output_dir: Directory to save processed audio files
        epsilon: L-infinity bound for the noise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all song folders
    song_folders = []
    for item in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, item)
        if os.path.isdir(folder_path):
            # Check if this folder contains perturbed_mixture.wav
            if os.path.exists(os.path.join(folder_path, "perturbed_mixture.wav")):
                song_folders.append(item)
    
    if not song_folders:
        print(f"No song folders with perturbed_mixture.wav found in {input_dir}")
        return
    
    print(f"Found {len(song_folders)} song folders to process")
    
    # Process each song folder
    results = []
    for song_name in tqdm(song_folders, desc="Processing songs"):
        try:
            # Define input and output paths
            input_path = os.path.join(input_dir, song_name, "perturbed_mixture.wav")
            output_path = os.path.join(output_dir, f"{song_name}.wav")
            
            # Load audio file
            audio, sample_rate = sf.read(input_path)
            
            # Add noise with L-infinity constraint
            perturbed_audio, noise = add_noise_with_linf_constraint(audio, epsilon)
            
            # Calculate Signal-to-Noise Ratio
            snr = calculate_snr(audio, noise)
            
            # Save perturbed audio
            sf.write(output_path, perturbed_audio, sample_rate)
            
            # Calculate statistics
            max_noise = np.max(np.abs(noise))
            l2_norm = np.sqrt(np.mean(noise ** 2))
            
            # Save result statistics
            results.append({
                'song_name': song_name,
                'max_noise': max_noise,
                'l2_norm': l2_norm,
                'snr': snr,
                'epsilon': epsilon
            })
            
            print(f"Processed {song_name}: SNR = {snr:.2f} dB")
            
        except Exception as e:
            print(f"Error processing {song_name}: {e}")
    
    # Save summary to CSV
    if results:
        df = pd.DataFrame(results)
        summary_path = os.path.join(output_dir, 'noise_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")
    
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description=
        'Process song folders containing perturbed_mixture.wav files by adding L-infinity bounded noise')
    
    # Input/output options
    parser.add_argument('--input-dir', required=True, 
                        help='Directory containing song folders with perturbed_mixture.wav files')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save processed audio files')
    
    # Noise parameters
    parser.add_argument('--epsilon', type=float, default=0.003,
                        help='L-infinity bound for noise (default: 0.01)')
    
    args = parser.parse_args()
    
    # Process all song folders
    process_song_folders(args.input_dir, args.output_dir, args.epsilon)

if __name__ == "__main__":
    main()