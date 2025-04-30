# -*- coding: utf-8 -*-
"""
Script to trim MUSDB18 .stem.mp4 files to extract the second 5-second segment.
Takes a folder of MUSDB18 songs and creates a new folder with 5-second versions.

Usage: python trim_musdb_stems.py input_folder output_folder
"""
import os
import sys
import subprocess
from tqdm import tqdm
import wave

def trim_stem_file(input_path, output_path, start_time=60, duration=90):
    """
    Trim a .stem.mp4 file to extract the second 5-second segment.
    
    Args:
        input_path: Path to the input .stem.mp4 file
        output_path: Path to save the trimmed .stem.mp4 file
        start_time: Start time in seconds (default: 5 seconds)
        duration: Duration in seconds to trim to (default: 5 seconds)
    """
    try:
        # Use FFmpeg to trim the file while preserving all tracks
        command = [
            "ffmpeg", 
            "-i", input_path,
            "-ss", str(start_time),  # Start at 5 seconds
            "-t", str(duration),     # Take 5 seconds
            "-c", "copy",  # Copy without re-encoding
            "-y",  # Overwrite output files
            output_path
        ]
        
        # Run FFmpeg and capture output
        result = subprocess.run(command, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        # Check if successful
        if result.returncode != 0:
            print(f"Error trimming {os.path.basename(input_path)}: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"Exception while processing {os.path.basename(input_path)}: {e}")
        return False

def trim_wav(input_path: str,
             start_sec: float = 0.0,
             end_sec: float = None,
             output_path: str = None) -> str:
    """
    Trim a WAV file without re-encoding.

    :param input_path:  Path to the .wav file.
    :param start_sec:   Start time in seconds (default=0.0).
    :param end_sec:     End time in seconds (if None, until EOF).
    :param output_path: Path for the trimmed file (if None, appends '_trimmed').
    :return:            The path to the trimmed file.
    :raises FileNotFoundError: If the input file doesn't exist.
    :raises ValueError: If end_sec <= start_sec.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No such file: {input_path}")
    song_dir_name = os.path.dirname(input_path)
    song_dir_name = os.path.basename(song_dir_name)
    channel_name = os.path.basename(input_path)
    if output_path is None:
        output_path = f"trimmed/{song_dir_name}/{channel_name}"

    # Open input WAV
    with wave.open(input_path, 'rb') as in_wav:
        params = in_wav.getparams()
        framerate = in_wav.getframerate()
        nframes = in_wav.getnframes()

        # Calculate frames to read
        start_frame = int(start_sec * framerate)
        end_frame = int(end_sec * framerate) if end_sec is not None else nframes
        if end_frame > nframes:
            end_frame = nframes
        if end_frame <= start_frame:
            raise ValueError("end_sec must be greater than start_sec")

        num_frames = end_frame - start_frame
        in_wav.setpos(start_frame)
        frames = in_wav.readframes(num_frames)

    # Write to output WAV
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with wave.open(output_path, 'wb') as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(frames)


def trim_each_song_channels(song_dir_path, start_sec, end_sec):
    for channel_name in ["vocals", "drums", "bass", "other","mixture"]:
        trim_wav(os.path.join(song_dir_path, f"{channel_name}.wav"), start_sec, end_sec)


def process_directory(input_dir, output_dir):
    """
    Process all .stem.mp4 files in a directory and trim each to the second 5 seconds.
    """
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of .stem.mp4 files in the input directory and subdirectories
    stem_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            # Check if it's a stem file
            if filename.endswith('.stem.mp4'):
                full_path = os.path.join(root, filename)
                # Get the relative path from input_dir
                rel_path = os.path.relpath(full_path, input_dir)
                stem_files.append((full_path, rel_path))
    
    if not stem_files:
        print(f"No .stem.mp4 files found in '{input_dir}' or its subdirectories.")
        return
    
    print(f"Found {len(stem_files)} .stem.mp4 files to process.")
    
    # Process each file with a progress bar
    for input_path, rel_path in tqdm(stem_files, desc="Trimming stem files"):
        # Create output path with same directory structure
        output_path = os.path.join(output_dir, rel_path)
        
        # Create any needed subdirectories
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Trim the file to extract the second 5-second segment
        success = trim_stem_file(input_path, output_path, start_time=0, duration=25)
        if not success:
            print(f"Failed to process {rel_path}")
    
    print(f"\nComplete! Processed {len(stem_files)} files from '{input_dir}' to '{output_dir}'")

def check_ffmpeg():
    """Check if FFmpeg is installed and available"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def get_duration(file_path):
    """Get the duration of a media file using FFprobe"""
    command = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        file_path
    ]
    
    try:
        result = subprocess.run(command, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    
    return None

if __name__ == "__main__":
    # Check if FFmpeg is installed
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed or not in the system PATH.")
        print("Please install FFmpeg before running this script.")
        sys.exit(1)

    # Check if both input and output directories are provided
    if len(sys.argv) != 3:
        print("Usage: python trim_musdb_stems.py input_folder output_folder")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(input_dir, output_dir)