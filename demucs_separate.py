import shutil
import subprocess
import os


def demucs_separate(input_song_path, estimation_output_path):
    """
    Runs Demucs to separate the audio file and moves the results directly to estimation_output_path.

    Parameters:
        input_song_path (str): Path to the input audio file.
        estimation_output_path (str): Directory to store the separated stems directly.
    """
    if not os.path.isfile(input_song_path):
        raise FileNotFoundError(f"Input file not found: {input_song_path}")

    # Ensure output directory exists
    os.makedirs(estimation_output_path, exist_ok=True)

    # Use Demucs model name
    model_name = "htdemucs"

    # Run Demucs separation
    command = [
        "python3", "-m", "demucs.separate",
        "-n", model_name,
        "--out", estimation_output_path,
        input_song_path
    ]

    try:
        subprocess.run(command, check=True)
        print("Separation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Demucs failed: {e}")
        return

    # Construct path to where Demucs actually put the results
    song_name = os.path.splitext(os.path.basename(input_song_path))[0]
    temp_output_dir = os.path.join(estimation_output_path, model_name, song_name)

    # Move files from temp_output_dir to estimation_output_path
    for file_name in os.listdir(temp_output_dir):
        src = os.path.join(temp_output_dir, file_name)
        dst = os.path.join(estimation_output_path, file_name)
        shutil.move(src, dst)

    # Clean up the intermediate folders created by Demucs
    shutil.rmtree(os.path.join(estimation_output_path, model_name))

    print(f"Results moved to: {estimation_output_path}")