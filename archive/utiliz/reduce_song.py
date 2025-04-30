# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:01:56 2025

@author: igor
"""

from pydub import AudioSegment

# Define the input and output file paths
input_file = "/../milk_cow/mixture.wav"   # Replace with your input file path
output_file = "song.wav" # Replace with your desired output file path

# Load the audio file (pydub supports many formats, such as wav, mp3, etc.)
audio = AudioSegment.from_file(input_file)

# Trim the audio to the first 15 seconds (15 seconds = 15000 milliseconds)
trimmed_audio = audio[:5000]

# Export the trimmed audio to a new file
trimmed_audio.export(output_file, format="wav")

print("Audio trimmed to 15 seconds and saved as", output_file)
