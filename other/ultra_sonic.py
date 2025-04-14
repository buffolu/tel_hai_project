import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment

def add_ultrasonic_noise(audio, sr=44100, duration=None, position=0, target="both", amplitude=0.001, boost_factor=10):
    """
    Add noise to the audio that is intended to change the waveform, the spectrogram, or both.
    
    Parameters:
      audio: A pydub AudioSegment object.
      sr: Sampling rate (default 44100 Hz).
      duration: Duration (in ms) for the noise. If None, uses the entire audio duration.
      position: Starting position (in ms) for the noise overlay.
      target: 'waveform', 'spectrogram', or 'both'
              - 'waveform': adds low-frequency noise (e.g., 1000 Hz) with boosted amplitude to visibly change the time-domain waveform.
              - 'spectrogram': adds ultrasonic noise (e.g., 25000 Hz) which can appear in the spectrogram when plotted over a wide frequency range.
              - 'both': combines the two noise types.
      amplitude: Base amplitude of the noise.
      boost_factor: Multiplier for the amplitude when targeting the waveform.
      
    Returns:
      An AudioSegment with the noise overlaid.
    """
    # Use entire audio duration if not specified (duration in ms)
    if duration is None:
        duration = len(audio)
    
    duration_sec = duration / 1000.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    
    # Initialize noise signal
    noise_signal = np.zeros_like(t)
    
    if target == "waveform":
        # Use a low frequency (e.g., 1000 Hz) to visibly alter the waveform.
        freq = 1000
        noise_signal = amplitude * boost_factor * np.sin(2 * np.pi * freq * t)
    elif target == "spectrogram":
        # Use ultrasonic frequency (e.g., 25000 Hz) for spectrogram changes.
        freq = 25000
        noise_signal = amplitude * np.sin(2 * np.pi * freq * t)
    elif target == "both":
        # Combine both effects: low frequency for the waveform and ultrasonic for the spectrogram.
        freq_waveform = 1000
        freq_spectro = 25000
        noise_waveform = amplitude * boost_factor * np.sin(2 * np.pi * freq_waveform * t)
        noise_spectro = amplitude * np.sin(2 * np.pi * freq_spectro * t)
        noise_signal = noise_waveform + noise_spectro
    
    # Convert noise_signal to 16-bit PCM values
    noise_int16 = np.int16(noise_signal * 32767)
    
    # Create an AudioSegment from the noise signal
    noise_audio = AudioSegment(
        noise_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16-bit audio
        channels=1
    )
    
    # Overlay the noise onto the original audio at the specified position
    combined_audio = audio.overlay(noise_audio, position=position)
    return combined_audio

def audiosegment_to_np(audio_seg):
    """
    Convert a pydub AudioSegment into a normalized NumPy array.
    """
    y = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    if audio_seg.channels > 1:
        y = y.reshape((-1, audio_seg.channels))
        y = y.mean(axis=1)
    # Normalize to [-1, 1]
    y = y / 32767.0
    return y

def plot_waveform_and_spectrogram(y, sr, title, ax_wave, ax_spec):
    """
    Plot a classic waveform (time vs. amplitude) and a spectrogram (frequency vs. time)
    on provided axes.
    """
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, alpha=0.7, ax=ax_wave)
    ax_wave.set(title=f"{title} Waveform", xlabel="Time (s)", ylabel="Amplitude")
    
    # Compute spectrogram using STFT (n_fft=2048 is common for a classic spectrogram)
    D = np.abs(librosa.stft(y, n_fft=2048))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    
    # Plot spectrogram with logarithmic frequency axis
    img = librosa.display.specshow(DB, sr=sr, x_axis="time", y_axis="log", cmap="magma", ax=ax_spec)
    ax_spec.set(title=f"{title} Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    return img

# ===== MAIN PROGRAM =====

# Ask the user for the audio file path
audio_path = input("Enter the audio file path: ").strip()

# Load the original audio using pydub (ensure you have ffmpeg installed if needed)
try:
    original_audio = AudioSegment.from_file(audio_path)
except Exception as e:
    print(f"Error loading audio: {e}")
    exit(1)

# Add noise to the audio; change the 'target' parameter as desired: "waveform", "spectrogram", or "both"
manipulated_audio = add_ultrasonic_noise(original_audio, sr=original_audio.frame_rate, target="both", amplitude=0.001, boost_factor=10)

# Save the manipulated audio to a new file (e.g., as WAV)
output_filename = "manipulated_version.wav"
manipulated_audio.export(output_filename, format="wav")
print(f"Manipulated audio saved as: {output_filename}")

# Convert both original and manipulated audio to NumPy arrays for plotting
y_original = audiosegment_to_np(original_audio)
y_manipulated = audiosegment_to_np(manipulated_audio)
sr = original_audio.frame_rate

# Create a 2x2 subplot grid:
# Top row: Original audio (left: waveform, right: spectrogram)
# Bottom row: Manipulated audio (left: waveform, right: spectrogram)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot original audio
img_orig = plot_waveform_and_spectrogram(y_original, sr, "Original", ax_wave=axes[0, 0], ax_spec=axes[0, 1])
fig.colorbar(img_orig, ax=axes[0, 1], format="%+2.0f dB")

# Plot manipulated audio
img_manip = plot_waveform_and_spectrogram(y_manipulated, sr, "Manipulated", ax_wave=axes[1, 0], ax_spec=axes[1, 1])
fig.colorbar(img_manip, ax=axes[1, 1], format="%+2.0f dB")

plt.tight_layout()
plt.show()
