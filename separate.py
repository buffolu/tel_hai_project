"""
Created on Sun Feb  9 17:05:03 2025

Modified to work with relative paths.
Each audio file's separated sources are saved in a subfolder (named after the audio file) 
inside the output folder to prevent overwriting.

@author: igor
"""

import os
import subprocess
import sys
import torch
import torchaudio
import shutil
from demucs.pretrained import get_model
from demucs.apply import apply_model

# ============================
# CONFIGURATION (Edit these if needed)
# ============================

# Relative path to the audio file or folder containing audio files
input_path = os.path.join(".", "defended")
#input_path = os.path.join(".", "song.wav")

# Relative path to the base output directory where separated sources will be saved
output_base_folder = os.path.join(".", "attack_separated")

# Select the Demucs model to use (e.g., "htdemucs", "htdemucs_ft", etc.)
selected_model = "htdemucs"

# Create output directory
os.makedirs(output_base_folder, exist_ok=True)

# Supported audio extensions
audio_extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Demucs model
model = get_model(selected_model)
model.to(device)

def process_file(file_path):
    """
    Process a single audio file using the selected Demucs model.
    Creates a subfolder named after the audio file in the output directory.
    """
    print(f"Processing: {file_path}...")
    
    # Get the base name of the file (without extension)
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create a dedicated folder for this song's outputs
    song_output_folder = os.path.join(output_base_folder, file_basename)
    os.makedirs(song_output_folder, exist_ok=True)
    
    # Copy the original audio to the song's output folder
    original_copy_path = os.path.join(song_output_folder, "mixture.wav")
    shutil.copy(file_path, original_copy_path)
    
    # Load audio file
    audio, sample_rate = torchaudio.load(file_path)
    
    # Ensure audio has correct shape [batch, channels, samples]
    if audio.dim() == 2:
        # [channels, samples] -> [1, channels, samples]
        audio = audio.unsqueeze(0)
    
    # Move audio to the same device as the model
    audio = audio.to(device)
    
    # Separate sources using the model
    with torch.no_grad():
        sources = apply_model(model, audio, device=device)
        
    # Sources shape: [batch, sources, channels, time]
    # Move back to CPU
    sources = sources.cpu()
    
    # Get source names from the model
    source_names = model.sources
    
    # Save each source as a separate file in the song's folder
    for i, source_name in enumerate(source_names):
        source_path = os.path.join(song_output_folder, f"{source_name}.wav")
        # Extract source and convert to [channels, samples] format for saving
        source_audio = sources[0, i]  # [channels, time]
        torchaudio.save(source_path, source_audio, sample_rate)
        print(f"Saved {source_name}.wav to {song_output_folder}")
    
    print(f"Successfully processed: {file_path}")
    print(f"All files saved to: {song_output_folder}\n")
    
    return sources

# Process the input (file or directory)
if os.path.isfile(input_path):
    if input_path.lower().endswith(audio_extensions):
        # For a single file, create a folder named after the file
        file_basename = os.path.splitext(os.path.basename(input_path))[0]
        process_file(input_path)
    else:
        print("The specified file does not appear to be a supported audio file.")
        sys.exit(1)
elif os.path.isdir(input_path):
    # Gather all supported audio files in the directory
    audio_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(audio_extensions) and os.path.isfile(os.path.join(input_path, f))
    ]
    
    if not audio_files:
        print(f"No audio files found in the directory '{input_path}'.")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio file(s) in the directory.\n")
    for file in audio_files:
        process_file(file)
else:
    print("The specified input path is neither a file nor a directory. Please check the path.")
    sys.exit(1)

print("Processing complete! All separated sources are saved in their respective folders in the output directory.")