# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:58:39 2025

Modified to ask the user for an input file or folder (supporting both WAV and MP3)
and for the output folder.
@author: igor
"""

import os
import random
import sys
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import WhiteNoise

def adjust_volume(audio, change_db):
    """Adjust the volume of the audio by the given dB change."""
    return audio + change_db


def add_subtle_noise(audio, noise_level=0.5, seed=None):
    """Add subtle white noise to the audio with less intensity."""
    if seed is not None:
        random.seed(seed)  # Set the seed for reproducibility
    
    noise = WhiteNoise().to_audio_segment(duration=len(audio))
    noise = noise - random.uniform(20, 40)  # Lower the noise level significantly
    return audio.overlay(noise)


def apply_random_volume_changes(audio, interval_ms=1000, min_db=-10, max_db=10, seed=None):
    """Apply random volume changes every second."""
    if seed is not None:
        random.seed(seed)  # Set the seed for reproducibility
    
    segments = []
    for start_ms in range(0, len(audio), interval_ms):
        end_ms = min(start_ms + interval_ms, len(audio))
        segment = audio[start_ms:end_ms]
        volume_change = random.uniform(min_db, max_db)
        segment = adjust_volume(segment, volume_change)
        segments.append(segment)
    return sum(segments)

def apply_band_masking(input_file, low_freq=1000, high_freq=3000):
    """Apply frequency band masking to an audio file."""
    y, sr = librosa.load(input_file, sr=None)
    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)

    # Define frequency band to mask
    freqs = librosa.fft_frequencies(sr=sr)
    mask_band = (freqs >= low_freq) & (freqs <= high_freq)

    # Apply masking
    magnitude[mask_band, :] = 0

    # Convert back to audio
    D_masked = magnitude * np.exp(1j * phase)
    y_masked = librosa.istft(D_masked)

    # Save the modified audio as WAV
    return y_masked,sr

def add_echo(audio, delay_ms=500, decay=0.5):
    """Adds an echo effect to the audio."""
    echo = audio._spawn(audio.raw_data)  # Create a duplicate of the audio
    echo = echo - (1 - decay) * 20  # Decay the echo by reducing its volume
    
    # Apply delay (create the echo by shifting the audio)
    echo = echo.set_frame_rate(audio.frame_rate)
    
    # Create the echo effect by overlaying the original and delayed versions
    audio_with_echo = audio.overlay(echo, position=delay_ms)
    
    return audio_with_echo

def add_closed_room_effect(audio, room_size=0.7, muffling=10, seed=None):
    """Simulates the sound of being in a closed room by adding muffling and reverb effects."""
    if seed is not None:
        random.seed(seed)  # Set the seed for reproducibility
    
    muffled_audio = audio.low_pass_filter(1500)  # Low pass filter to simulate muffled sound
    reverb_audio = muffled_audio + random.uniform(-room_size, room_size) * 5  # Slight reverb effect
    return reverb_audio
def add_ultrasonic_frequencies(audio, sr=44100, frequencies=[21000, 22000, 23000, 24000, 25000], duration=None, position=0, amplitude=1.0):
    """
    Add a very big ultrasonic noise to the audio by summing multiple ultrasonic sine waves.
    
    Parameters:
      audio: A pydub AudioSegment object.
      sr: Sampling rate (default 44100 Hz).
      frequencies: List of ultrasonic frequencies to include (all above 20 kHz).
      duration: Duration in ms for the noise. If None, uses the entire audio duration.
      position: Starting position (in ms) where the noise is overlaid.
      amplitude: Amplitude for the noise (default 1.0 for maximum effect).
      
    Returns:
      A new AudioSegment with the ultrasonic noise overlaid.
    """
    # Use entire audio duration if not specified
    if duration is None:
        duration = len(audio)  # duration in milliseconds
    
    duration_sec = duration / 1000.0  # convert ms to seconds
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    
    # Create ultrasonic noise by summing multiple sine waves
    ultrasonic_noise = np.zeros_like(t)
    for freq in frequencies:
        ultrasonic_noise += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Normalize the noise so that the amplitude does not exceed bounds
    ultrasonic_noise = ultrasonic_noise / len(frequencies)
    
    # Convert the noise array to 16-bit PCM data
    ultrasonic_int16 = np.int16(ultrasonic_noise * 32767)
    
    # Create an AudioSegment from the ultrasonic noise
    ultrasonic_audio = AudioSegment(
        ultrasonic_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16-bit audio
        channels=1
    )
    
    # Overlay the ultrasonic noise onto the original audio at the specified position
    combined_audio = audio.overlay(ultrasonic_audio, position=position)
    
    return combined_audio

def add_infrasonic_frequencies(audio, sr=44100, frequency=10, duration=None, position=0, amplitude=0.001):
    """
    Add infrasonic frequencies (below 20 Hz) to the audio in a way that is intended to be inaudible.
    """
    if duration is None:
        duration = len(audio)
    duration_sec = duration / 1000.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    infrasonic_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    infrasonic_int16 = np.int16(infrasonic_wave * 32767)
    infrasonic_audio = AudioSegment(
        infrasonic_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16-bit audio
        channels=1
    )
    combined_audio = audio.overlay(infrasonic_audio, position=position)
    return combined_audio



def process_audio(input_file, output_folder, manipulation, noise_level=0.05, echo_delay_ms=500, echo_decay=0.5, room_size=0.7, muffling=10, frequency=25000, duration=2000,seed = None):
    """Process the audio file based on the selected manipulation."""
    # Determine the file extension and use it to load the file
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext not in [".wav", ".mp3"]:
        print(f"Unsupported file format: {file_ext}. Only WAV and MP3 are supported.")
        return

    # Load the audio file
    audio = AudioSegment.from_file(input_file, format=file_ext[1:])
    
    # Extract the actual song_name from the parent folder
    song_folder = os.path.dirname(input_file)
    song_name = os.path.basename(os.path.dirname(song_folder))  # Corrected to get the song name

    # Determine if input is from "originals" or "estimated"
    input_folder = os.path.basename(song_folder)  # Either "originals" or "estimated"
    
    # Define the folder for each manipulation type (without song_name folder)
    manipulation_folder = os.path.join(output_folder, f"{song_name}_{manipulation}")

    # Based on input folder (originals/estimated), define the output subfolder
    if input_folder == "originals":
        output_folder_name = "originals"
    elif input_folder == "estimated":
        output_folder_name = "estimated"
    else:
        print("Unknown folder structure, expected either 'originals' or 'estimated'.")
        return

    # Create the manipulation folder inside the output folder
    os.makedirs(manipulation_folder, exist_ok=True)  # Create manipulation folder (e.g., song_name_band_mask)

    # Create the output folder for originals/estimated inside that manipulation folder
    output_path = os.path.join(manipulation_folder, output_folder_name)
    os.makedirs(output_path, exist_ok=True)  # Create originals/estimated folder if not exists

    # Create a function to handle saving the file for any manipulation
    def save_file(manipulated_audio):
        # Save the manipulated audio file
        output_filename = os.path.basename(input_file)
        manipulated_audio.export(os.path.join(output_path, output_filename), format="wav")
        print(f"Processed file saved as: {os.path.join(output_path, output_filename)}")

    # Apply the selected manipulation
    if manipulation == "volume":
        manipulated_audio = apply_random_volume_changes(audio,seed)
        save_file(manipulated_audio)
    elif manipulation == "noise":
        manipulated_audio = add_subtle_noise(audio, noise_level,seed)
        save_file(manipulated_audio)
    elif manipulation == "band_mask":
        # Apply band masking with librosa and save the result as a .wav file
        y_masked, sr = apply_band_masking(input_file)
        sf.write(os.path.join(output_path, os.path.basename(input_file)), y_masked, sr)
        print(f"Band-masked file saved as: {os.path.join(output_path, os.path.basename(input_file))}")
    elif manipulation == "echo":
        manipulated_audio = add_echo(audio, delay_ms=echo_delay_ms, decay=echo_decay)
        save_file(manipulated_audio)
    elif manipulation == "closed_room":
        manipulated_audio = add_closed_room_effect(audio, room_size=room_size, muffling=muffling,seed=seed)
        save_file(manipulated_audio)
    elif manipulation == "ultrasonic":
        manipulated_audio = add_ultrasonic_frequencies(audio, duration=duration)
        save_file(manipulated_audio)
    elif manipulation == "infrasonic":
        manipulated_audio = add_infrasonic_frequencies(audio, frequency=frequency, duration=duration)
        save_file(manipulated_audio)
    elif manipulation == "all":
        # Process all manipulations, creating separate folders for each manipulation
        manipulations = ["volume", "noise", "band_mask", "echo", "closed_room", "ultrasonic", "infrasonic"]
        for manip in manipulations:
            manip_folder = os.path.join(output_folder, f"{song_name}_{manip}")  # Make unique folder for each manipulation
            os.makedirs(manip_folder, exist_ok=True)  # Create a folder for each manipulation type

            output_subfolder = os.path.join(manip_folder, "originals")  # Create the "originals" subfolder for each manipulation
            os.makedirs(output_subfolder, exist_ok=True)

            # Process based on the manipulation type
            if manip == "volume":
                manipulated_audio = apply_random_volume_changes(audio, seed=seed)
            elif manip == "noise":
                manipulated_audio = add_subtle_noise(audio, noise_level, seed=seed)
            elif manip == "band_mask":
                y_masked, sr = apply_band_masking(input_file)
                sf.write(os.path.join(output_subfolder, os.path.basename(input_file)), y_masked, sr)
                print(f"Band-masked file saved as: {os.path.join(output_subfolder, os.path.basename(input_file))}")
                continue  # Skip file export for band_mask, it's already handled
            elif manip == "echo":
                manipulated_audio = add_echo(audio, delay_ms=echo_delay_ms, decay=echo_decay)
            elif manip == "closed_room":
                manipulated_audio = add_closed_room_effect(audio, room_size=room_size, muffling=muffling, seed=seed)
            elif manip == "ultrasonic":
                manipulated_audio = add_ultrasonic_frequencies(audio, duration=duration)
            elif manip == "infrasonic":
                manipulated_audio = add_infrasonic_frequencies(audio, frequency=frequency, duration=duration)

            # Save the manipulated file in the correct folder
            output_filename = os.path.basename(input_file)
            manipulated_audio.export(os.path.join(output_subfolder, output_filename), format="wav")
            print(f"Processed file saved as: {os.path.join(output_subfolder, output_filename)}")

        return  # After processing all, return to avoid redundant exports



    # Export the processed audio for non-"all" manipulations
    save_file(audio)  # For non-"all" manipulations





def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Prompt the user for an input path (file or directory)
    input_path = input("Enter the full path to a WAV or MP3 file or a directory containing them: ").strip()
    if not os.path.exists(input_path):
        print("Error: The specified input path does not exist.")
        return
    
    # Prompt the user for an output folder
    output_folder = input("Enter the output folder path for manipulated audio files: ").strip()
    if not os.path.isabs(output_folder):
        output_folder = os.path.join(script_directory, output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    #apply seed
    print("apply seed?")
    print("1. yes")
    print("2. no")
    seed =  input("Enter the number corresponding to your choice: ").strip()

    if seed not in ["1", "2"]:
        print("Invalid choice. Please select a valid option.")
        return
    if seed == "2":
        seed = None
    if seed == "1":
        seed = 42
    
    # Present manipulation choices
    print("\nSelect an audio manipulation:")
    print("1. Volume Adjustment")
    print("2. Add Subtle Noise")
    print("3. Apply Band Masking")
    print("4. Apply Echo Effect")
    print("5. Apply Closed Room Effect")
    print("6. Apply Ultrasonic Frequencies")
    print("7. Apply Infrasonic Frequencies")
    print("8. Apply All Manipulations")
    choice = input("Enter the number corresponding to your choice: ").strip()
    
    if choice not in ["1", "2", "3", "4", "5", "6", "7", "8"]:
        print("Invalid choice. Please select a valid option.")
        return
    
    manipulation_map = {
        "1": "volume",
        "2": "noise",
        "3": "band_mask",
        "4": "echo",
        "5": "closed_room",
        "6": "ultrasonic",
        "7": "infrasonic",
        "8": "all"
    }
    manipulation = manipulation_map[choice]
    
    # Process based on whether the input path is a file or directory
    audio_extensions = (".wav", ".mp3")
    if os.path.isfile(input_path):
        if input_path.lower().endswith(audio_extensions):
            process_audio(input_path, output_folder, manipulation,seed=seed)
        else:
            print("The specified file is not a supported audio file (WAV or MP3).")
    elif os.path.isdir(input_path):
        audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(audio_extensions)]
        if not audio_files:
            print("No WAV or MP3 files found in the specified directory.")
            return
        for filename in audio_files:
            file_path = os.path.join(input_path, filename)
            try:
                process_audio(file_path, output_folder, manipulation,seed=seed)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    else:
        print("The specified input is neither a file nor a directory.")

if __name__ == "__main__":
    main()
