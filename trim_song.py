import subprocess
import argparse
import os
import wave
import stempeg

def separate_stem_song_into_channels(input_path, directory_path):
    stem_names = ['mixture', 'drums', 'bass', 'other', 'vocals']
    audio, rate = stempeg.read_stems(input_path, stem_id=None)

    for i, stem_audio in enumerate(audio):
        print(f"{stem_names[i]} shape: {stem_audio.shape}")
        output_path = os.path.join(directory_path, f"{stem_names[i]}.wav")
        stempeg.write_audio(output_path, stem_audio, rate)
def trim_stem_file(input_path, output_path, start_time, duration):
    """
    Trim a .stem.mp4 file while preserving all audio channels.

    Parameters:
    - input_path: Path to input .stem.mp4 file
    - output_path: Path to save the trimmed file
    - start_time: Start time in seconds (or format "00:00:05.0")
    - duration: Duration in seconds (or format "00:00:10.0")
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # FFmpeg command to trim the file without re-encoding (-c copy)
    command = [
        "ffmpeg",
        "-y",                        # Overwrite output file if it exists
        "-i", input_path,           # Input file
        "-ss", str(start_time),     # Start time
        "-t", str(duration),        # Duration
        "-c", "copy",               # Copy codec to preserve stems
        output_path
    ]

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)
    print(f"Trimmed stem file saved to: {output_path}")

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


def trim_song_channels(originals_path,trimmed_path,start_sec, end_sec):
    for channel_name in ["vocals", "drums", "bass", "other","mixture"]:
        channel_wav_file_path = os.path.join(originals_path, f"{channel_name}.wav")
        trimmed_channel_wav_file_path = os.path.join(trimmed_path, f"{channel_name}.wav")
        trim_wav(channel_wav_file_path, start_sec, end_sec,trimmed_channel_wav_file_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim a .stem.mp4 file without damaging channel separation.")
    parser.add_argument("song_name", help="song name")
    parser.add_argument("--start", type=int, required=True, help="Start time (e.g., 00:00:10 or 10)")
    parser.add_argument("--duration",type=int, required=True, help="Trim duration (e.g., 00:00:30 or 30)")
    args = parser.parse_args()


    song_stem_path = os.path.join("songs", args.song_name, "original_song", f"{args.song_name}.stem.mp4")
    original_song_path = os.path.join("songs", args.song_name, "original_song")
    separate_stem_song_into_channels(song_stem_path, original_song_path)

    output = os.path.join("songs", args.song_name, "trimmed_song")
    os.makedirs(output, exist_ok=True)

    trim_song_channels(original_song_path, output, args.start, args.duration)
