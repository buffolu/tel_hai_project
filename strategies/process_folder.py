# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 14:38:14 2025

@author: igor
"""

#!/usr/bin/env python3
"""
Batch processor for running Yao_Qin_2019.py on all audio files in a folder.
Usage: python process_folder.py /path/to/songs_test
"""

import os
import sys
import subprocess
import argparse
from tqdm import tqdm

def process_audio_folder(input_folder, script_path="strategies/Yao_Qin_2019.py", additional_args=None):
    """
    Process all audio files in the given folder using the specified script.
    
    Args:
        input_folder: Path to the folder containing audio files
        script_path: Path to the processing script
        additional_args: Additional arguments to pass to the script
    """
    # Check if folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        return
    
    # Create a list of supported audio extensions
    audio_extensions = ['.mp3','.mp4', '.wav', '.flac', '.ogg', '.m4a']
    
    # Get all audio files in the folder
    audio_files = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Skip directories and non-audio files
        if os.path.isdir(file_path):
            continue
            
        # Check if the file has a supported audio extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in audio_extensions:
            continue
            
        audio_files.append((filename, file_path))
    
    total_files = len(audio_files)
    if total_files == 0:
        print(f"No audio files found in '{input_folder}'")
        return
    
    print(f"Found {total_files} audio files to process")
    
    # Process each file with a progress bar
    for i, (filename, file_path) in enumerate(audio_files):
        # Construct the command
        cmd = ["python", script_path, "--input_file", file_path, "--save_sources"]
        
        # Add any additional arguments
        if additional_args:
            cmd.extend(additional_args)
            
        # Print the command being executed
        print(f"\n[{i + 1}/{total_files}] Processing: {filename}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Execute the command
            subprocess.run(cmd, check=True)
            print(f"✓ Successfully processed: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {filename}: {e}")
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            return
        except Exception as e:
            print(f"✗ Unexpected error processing {filename}: {e}")
    
    # Print summary
    print(f"\nProcessing complete. Processed {total_files} files.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Batch process audio files with Yao_Qin_2019.py")
    parser.add_argument("input_folder", help="Folder containing audio files to process")
    parser.add_argument("--script", default="Yao_Qin_2019.py", help="Path to the processing script")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional arguments to pass to the script")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the folder
    process_audio_folder(args.input_folder, args.script, args.extra_args)

if __name__ == "__main__":
    main()