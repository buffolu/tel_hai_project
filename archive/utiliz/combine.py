import os
from pydub import AudioSegment

def overlay_audio_files(input_folder, output_file):
    """Overlay all audio files in the given folder to create one mixed track."""
    # Get list of audio files (handle case differences in extension)
    audio_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".wav", ".mp3"))
    ]
    
    if not audio_files:
        print("No audio files found in the folder.")
        return

    # Load the first file as the base track
    combined_audio = AudioSegment.from_file(audio_files[0])
    print(f"Using {os.path.basename(audio_files[0])} as base track.")

    # Overlay the rest of the audio files
    for file_path in audio_files[1:]:
        print(f"Overlaying {os.path.basename(file_path)}...")
        next_audio = AudioSegment.from_file(file_path)
        combined_audio = combined_audio.overlay(next_audio)

    # Export the combined audio
    combined_audio.export(output_file, format="wav")
    print(f"Combined (overlaid) audio saved as: {output_file}")

# Ask user for input folder and output file location
input_folder = input("Enter the folder path containing audio files: ").strip()
output_file = input("Enter the path and name for the output file (e.g., output.wav): ").strip()

if not os.path.exists(input_folder):
    print("The specified folder does not exist. Exiting.")
else:
    overlay_audio_files(input_folder, output_file)
