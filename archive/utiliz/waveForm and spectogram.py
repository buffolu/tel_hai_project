import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_audio(file_path):
    """Load an audio file using librosa."""
    return librosa.load(file_path, sr=None)

# Ask user for multiple audio file paths
audio_files = []
print("Enter audio file paths (type 'done' when finished):")
while True:
    file_path = input("Enter file path: ").strip()
    if file_path.lower() == "done":
        break
    audio_files.append(file_path)

num_songs = len(audio_files)

# Create a figure with num_songs rows and 2 columns
fig, axes = plt.subplots(num_songs, 2, figsize=(14, 4 * num_songs))

# Ensure axes is 2D even when there's only one song
if num_songs == 1:
    axes = np.array([axes])

for i, file_path in enumerate(audio_files):
    try:
        # Load the audio file
        y, sr = load_audio(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

    # Left column: Plot the classic waveform
    librosa.display.waveshow(y, sr=sr, alpha=0.7, ax=axes[i, 0])
    axes[i, 0].set_title(f"Waveform - {file_path}")
    axes[i, 0].set_xlabel("Time (s)")
    axes[i, 0].set_ylabel("Amplitude")

    # Right column: Compute and plot the classic spectrogram
    # Standard STFT settings (n_fft=2048 is common for a classic view)
    D = np.abs(librosa.stft(y, n_fft=2048))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    img = librosa.display.specshow(DB, sr=sr, x_axis="time", y_axis="log",
                                   cmap="magma", ax=axes[i, 1])
    axes[i, 1].set_title(f"Spectrogram - {file_path}")
    axes[i, 1].set_xlabel("Time (s)")
    axes[i, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=axes[i, 1], format="%+2.0f dB")

plt.tight_layout()
plt.show()
