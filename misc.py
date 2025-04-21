# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:25:53 2025

@author: igor
"""

#!/usr/bin/env python3
"""
Audio Information Tool

This script analyzes audio files and displays detailed information about
their length, channels, sample rate, and other properties.

Usage:
    python audio_info.py <path_to_audio_file>
"""

import sys
import os
import torch
import torchaudio
import argparse
from pathlib import Path
import numpy as np

def print_audio_length(audio, sample_rate=44100, name="Audio"):
    """
    Print detailed information about audio length in both samples and seconds.
    
    Args:
        audio: Audio tensor (torch tensor)
        sample_rate: Audio sample rate in Hz
        name: Name to display for this audio
    """
    # Handle shape based on tensor dimensions
    if audio.dim() == 1:  # Mono [samples]
        channels = 1
        samples = audio.shape[0]
    elif audio.dim() == 2:  # Standard [channels, samples] PyTorch format
        channels = audio.shape[0]
        samples = audio.shape[1]
    elif audio.dim() == 3:  # Batched audio [batch, channels, samples]
        batch = audio.shape[0]
        channels = audio.shape[1]
        samples = audio.shape[2]
        print(f"{name}: [Batch of {batch}]")
    else:
        print(f"{name} has unusual shape: {audio.shape}")
        return
    
    # Calculate duration
    duration = samples / sample_rate
    
    # Format duration as minutes:seconds
    minutes = int(duration // 60)
    seconds = duration % 60
    time_str = f"{minutes}:{seconds:06.3f}"
    
    # Print information
    print(f"\n{name}:")
    print(f"  Shape: {audio.shape}")
    print(f"  Channels: {channels}")
    print(f"  Samples: {samples:,}")
    print(f"  Duration: {duration:.4f} seconds ({time_str})")
    print(f"  Sample Rate: {sample_rate:,} Hz")
    
    if hasattr(audio, 'device'):
        print(f"  Device: {audio.device}")
    
    # Get min/max values
    with torch.no_grad():
        min_val = audio.min().item()
        max_val = audio.max().item()
        rms = torch.sqrt(torch.mean(audio**2)).item()
        
        print(f"  Value Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  RMS Value: {rms:.4f}")
        
        # Check if values are normalized
        if max(abs(min_val), abs(max_val)) <= 1.0:
            print("  Normalization: Values appear to be normalized (-1 to 1)")
        else:
            print("  Normalization: Values do NOT appear to be normalized")
            
        # Print dB information
        if max(abs(min_val), abs(max_val)) > 0:
            peak_db = 20 * np.log10(max(abs(min_val), abs(max_val)))
            rms_db = 20 * np.log10(rms) if rms > 0 else -float('inf')
            print(f"  Peak Level: {peak_db:.2f} dB")
            print(f"  RMS Level: {rms_db:.2f} dB")

def analyze_audio_file(file_path):
    """
    Analyze an audio file and print detailed information.
    
    Args:
        file_path: Path to the audio file
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        print(f"Error: File does not exist: {file_path}")
        return 1
        
    # Get file information
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print("\nFile Information:")
    print(f"  Path: {file_path.absolute()}")
    print(f"  Size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    print(f"  Extension: {file_path.suffix}")
    
    try:
        # Load audio
        audio, sample_rate = torchaudio.load(str(file_path))
        
        # Print audio information
        file_name = file_path.name
        print_audio_length(audio, sample_rate, f"Audio: {file_name}")
        
        # Additional useful info for debugging:
        
        # Check for non-finite values
        non_finite = torch.sum(~torch.isfinite(audio)).item()
        if non_finite > 0:
            print(f"  WARNING: Found {non_finite} non-finite values (NaN/Inf)")
        
        # Check for silence or very quiet segments
        is_silent = torch.all(torch.abs(audio) < 1e-6).item()
        if is_silent:
            print("  WARNING: Audio appears to be silent (all values near zero)")
            
        # Check for clipping
        clipped_samples = torch.sum((torch.abs(audio) > 0.99)).item()
        if clipped_samples > 0:
            clip_percent = 100 * clipped_samples / (audio.shape[0] * audio.shape[1])
            print(f"  WARNING: Possible clipping detected ({clip_percent:.4f}% of samples)")
        
        # Check if audio length is divisible by common factors
        # This can help identify if length issues might be related to model constraints
        print("\nLength Analysis:")
        for factor in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            remainder = audio.shape[1] % factor
            if remainder == 0:
                print(f"  Audio length is divisible by {factor}")
            else:
                print(f"  Audio length / {factor} has remainder {remainder}")
                
        return 0
                
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(description="Analyze audio files and display detailed information")
    parser.add_argument("file_path", help="Path to the audio file")
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
        
    args = parser.parse_args()
    return analyze_audio_file(args.file_path)

if __name__ == "__main__":
    sys.exit(main())