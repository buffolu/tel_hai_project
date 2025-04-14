# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:27:14 2025

@author: igor
"""
import os
from pydub import AudioSegment
import sys

def convert_m4a_to_wav(input_folder, output_folder):
    """Converts all .m4a files in the input folder to .wav format and stores them in the output folder."""
    # Ensure the output folder exists; create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".m4a"):
            m4a_path = os.path.join(input_folder, filename)
            wav_filename = f"{os.path.splitext(filename)[0]}.wav"
            wav_path = os.path.join(output_folder, wav_filename)
            try:
                audio = AudioSegment.from_file(m4a_path, format="m4a")
                audio.export(wav_path, format="wav")
                print(f"Converted {filename} to {wav_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

def main():
    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Define the input and output folders
    input_ = sys.argv[1]
    output_ = sys.argv[2]
    
    
    input_folder = os.path.join(script_directory, input_)  # Input folder named 'songs'
    output_folder = os.path.join(script_directory, output_)  # Output folder named 'converted_wav_files'
    
    # Convert the files
    convert_m4a_to_wav(input_folder, output_folder)

if __name__ == "__main__":
    main()

