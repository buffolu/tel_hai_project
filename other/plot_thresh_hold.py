# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:39:26 2025

@author: igor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
import librosa
import os

def calculate_masking_threshold(audio, sr, frame_size=2048, hop_length=512):
    """
    Calculate a simplified psychoacoustic masking threshold based on the audio spectrum.
    
    Parameters:
    -----------
    audio : numpy array
        The audio signal
    sr : int
        Sample rate
    frame_size : int
        FFT frame size
    hop_length : int
        Hop length for STFT
        
    Returns:
    --------
    frequencies : numpy array
        Frequency bins
    spectrum : numpy array
        Power spectrum of the audio
    masking_threshold : numpy array
        Estimated masking threshold
    """
    # Calculate the STFT of the audio
    S = librosa.stft(audio, n_fft=frame_size, hop_length=hop_length)
    
    # Convert to power spectrum
    power_spectrum = np.abs(S) ** 2
    
    # Average the power spectrum across time frames
    avg_spectrum = np.mean(power_spectrum, axis=1)
    
    # Convert to dB scale
    eps = 1e-10  # To avoid log(0)
    spectrum_db = 10 * np.log10(avg_spectrum + eps)
    
    # Calculate frequency bins
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    
    # Simple spreading function to simulate masking effect
    # This is a simplified version of what happens in actual psychoacoustic models
    masking_threshold_db = np.zeros_like(spectrum_db)
    
    # Parameters for the spreading function
    spread_lower = 3  # How many bins to spread downward (lower frequencies)
    spread_upper = 8  # How many bins to spread upward (higher frequencies)
    
    # Apply spreading function for each frequency bin
    for i in range(len(spectrum_db)):
        # Define the range of affected bins
        lower_bound = max(0, i - spread_lower)
        upper_bound = min(len(spectrum_db), i + spread_upper)
        
        # Contribution to masking threshold from this bin
        for j in range(lower_bound, upper_bound):
            # The masking effect decreases with distance from the masker
            distance = abs(i - j)
            # The decrease is steeper for lower frequencies masking higher ones
            if j > i:  # Higher frequency than masker
                decrease = 15 + 10 * distance  # Steeper decrease
            else:  # Lower frequency than masker
                decrease = 10 + 3 * distance   # Less steep decrease
                
            # Calculate the masking contribution
            masking_contribution = spectrum_db[i] - decrease
            
            # Update the masking threshold if this contribution is higher
            masking_threshold_db[j] = max(masking_threshold_db[j], masking_contribution)
    
    # Absolute threshold of hearing (simplified)
    absolute_threshold = np.ones_like(frequencies) * -80  # dB
    for i, freq in enumerate(frequencies):
        if freq < 100:
            absolute_threshold[i] = -40
        elif freq < 500:
            absolute_threshold[i] = -60
        elif freq > 10000:
            absolute_threshold[i] = -60
    
    # The final masking threshold is the maximum of the masking threshold and the absolute threshold
    masking_threshold_db = np.maximum(masking_threshold_db, absolute_threshold)
    
    # Add some offset to make sure we're safely below the audible level
    masking_threshold_db -= 10
    
    return frequencies, spectrum_db, masking_threshold_db

def visualize_masking(audio_file):
    """
    Visualize the audio spectrum and masking threshold for an audio file.
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # If stereo, convert to mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    
    # Calculate the masking threshold
    frequencies, spectrum, masking_threshold = calculate_masking_threshold(y, sr)
    
    # Plot the results
    plt.figure(figsize=(12, 7))
    
    # Plot the audio spectrum
    plt.plot(frequencies, spectrum, 'b-', linewidth=2, label='Original Audio')
    
    # Plot the masking threshold
    plt.plot(frequencies, masking_threshold, 'r--', linewidth=2, label='Masking Threshold')
    
    # Add a potential adversarial perturbation (just for illustration)
    # In a real attack, this would be calculated via optimization
    perturbation = masking_threshold - np.random.uniform(1, 3, size=len(masking_threshold))
    plt.plot(frequencies, perturbation, 'g-', linewidth=1, label='Example Perturbation')
    
    # Format the plot
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Audio Spectrum and Psychoacoustic Masking Threshold')
    plt.xlim(20, sr/2)  # From 20 Hz to Nyquist frequency
    plt.ylim(min(masking_threshold) - 10, max(spectrum) + 10)
    plt.legend()
    
    # Add vertical lines at common frequencies
    for f in [100, 500, 1000, 5000, 10000]:
        if f < sr/2:
            plt.axvline(x=f, color='gray', linestyle=':', alpha=0.5)
            plt.text(f, min(masking_threshold) - 5, f'{f} Hz' if f < 1000 else f'{f/1000}kHz',
                    horizontalalignment='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.splitext(os.path.basename(audio_file))[0] + '_masking.png'
    plt.savefig(output_file, dpi=300)
    plt.show()
    
    print(f"Plot saved as {output_file}")
    
    # Return frequencies and thresholds for potential further analysis
    return frequencies, spectrum, masking_threshold

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python psychoacoustic_masking.py mixture.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    visualize_masking(audio_file)