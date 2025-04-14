# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:33:36 2025

@author: igor
"""


import numpy as np
import os
import torch
import torch.nn.functional as F
import torchaudio
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from demucs.pretrained import get_model
#from create_threshold import PsychoacousticMasker, FrequencyDomainTransform
from demucs.apply import apply_model
import sys





def compute_stage1_loss(model, perturbed_audio, target_sources, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute loss for stage 1 attack - focus on degrading separation quality.
    Fixed to ensure device consistency.
    
    Args:
        model: Demucs model instance
        perturbed_audio: Perturbed audio tensor
        target_sources: Dictionary of original source tensors
        device: Computation device
        
    Returns:
        torch.Tensor: Loss value (negative for maximization)
    """
    
    # Ensure perturbed audio is on the correct device
    perturbed_audio = perturbed_audio.to(device)
    
    # Extract source tensors from dictionary, excluding mixture
    source_tensors = []
    for key, source in target_sources.items():
        if key != 'mixture':
            # Ensure each source is on the correct device
            source_tensors.append(source.to(device))
    
    # Ensure perturbed_audio has batch dimension
    if perturbed_audio.dim() == 2:
        perturbed_audio = perturbed_audio.unsqueeze(0)
    
    # Forward pass through Demucs
    estimated_sources = apply_model(model, perturbed_audio, device=device)
    estimated_sources = estimated_sources.squeeze(0)  # Now shape [4, 2, samples]
    
    # Calculate spectral loss for each source and scale
    def spectral_loss(estimated, target, scales=[4096, 2048, 1024, 512, 256], device=None):
        if device is None:
            device = estimated.device
        
        loss = 0
        for scale in scales:
            # Ensure scale doesn't exceed signal length
            if estimated.shape[-1] < scale:
                continue
                
            # Compute STFT for both signals
            est_spec = torch.stft(
                estimated, 
                n_fft=scale, 
                hop_length=scale//4,
                window=torch.hann_window(scale).to(device),
                return_complex=True
            )
            
            target_spec = torch.stft(
                target, 
                n_fft=scale, 
                hop_length=scale//4,
                window=torch.hann_window(scale).to(device),
                return_complex=True
            )
            
            # Compute L1 loss on magnitude spectrograms
            loss += F.l1_loss(torch.abs(est_spec), torch.abs(target_spec), reduction='mean')
            
        return loss

    def time_loss(estimated, target):
        return F.l1_loss(estimated, target, reduction='mean')
    
    # Compute total loss across all sources
    total_loss = 0
    for i, (est_source, target_source) in enumerate(zip(estimated_sources, source_tensors)):
        # Compute losses for this source
        spec_loss_val = spectral_loss(est_source, target_source, device=device)
        time_loss_val = time_loss(est_source, target_source)
        
        # Combine with weights
        source_loss = 0.8 * spec_loss_val + 0.2 * time_loss_val
        total_loss += source_loss
    
    # For adversarial attack, we want to maximize the loss (minimize quality)
    # So we return negative loss (higher values = better attack)
    return -total_loss

"""
def compute_masking_threshold(audio, window_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
    """"""
    Compute the psychoacoustic masking threshold using the PsychoacousticMasker.
    Adapts the NumPy implementation to work with PyTorch tensors for gradient computation.
    
    Args:
        audio: Tensor of shape [1, samples] or [samples] - mono audio
        window_size: Window size for STFT
        device: Device to use for computation
        
    Returns:
        masking_threshold: Tensor of shape [freq_bins, time_frames] - the masking threshold
    """"""
    # Convert tensor to NumPy for the PsychoacousticMasker
    audio_np = audio.cpu().detach().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    
    # Create masker instance
    masker = PsychoacousticMasker(window_size=window_size, sample_rate=44100)
    
    # Compute masking thresholds
    theta_xs, psd_max, freqs = masker.generate_th(audio_np)
    
    # Convert back to tensor
    masking_threshold = torch.tensor(theta_xs.T, dtype=torch.float32, device=device)
    
    return masking_threshold, psd_max

def compute_psychoacoustic_loss(perturbation, original_audio, window_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
    """""""
    Compute the psychoacoustic masking loss in a differentiable way.
    Measures how much the perturbation exceeds the masking threshold.
    """""""
    if perturbation.dim() == 2:
        perturbation = perturbation.unsqueeze(0)
    
    if original_audio.dim() == 2:
        original_audio = original_audio.unsqueeze(0)
    
    batch_size, channels, samples = perturbation.shape
    total_loss = 0
    for b in range(batch_size):
        for c in range(channels):
            # Get the channel data
            orig_audio_channel = original_audio[b, c].unsqueeze(0)

            pert_channel = perturbation[b, c].unsqueeze(0)
            
            # Compute masking threshold for this channel
            masking_threshold, psd_max = compute_masking_threshold(
                orig_audio_channel, window_size, device
            )
            # Use torch.stft with settings that match the original code
            # Using default hop_length to match librosa.core.stft default
            pert_stft = torch.stft(
                pert_channel,
                n_fft=window_size,
                hop_length=512,  # Default hop length in librosa
                window=torch.hann_window(window_size).to(device),
                return_complex=True,
                center=False  # Match the original setting
            )
            pert_stft = pert_stft.squeeze()
            pert_magnitude = torch.abs(pert_stft) / window_size
            pert_magnitude_squared = pert_magnitude**2
          
            # Get max value for normalization
            pert_psd_db = 10 * torch.log10(pert_magnitude_squared + 1e-20)
            
            # IMPORTANT FIX: Normalize to match masking threshold scale
            pert_psd = 96 - torch.max(pert_psd_db) + pert_psd_db
            
            # Debug after fixing normalization
            violation_pre_relu = pert_psd - masking_threshold
            print(f"After normalization fix:")
            print(f"  Max potential violation: {violation_pre_relu.max().item()}")
            print(f"  Mean potential violation: {violation_pre_relu.mean().item()}")
            print(f"  % of bins with violation: {(violation_pre_relu > 0).float().mean().item() * 100}%")
            
            # Calculate exceedance with margin to ensure learning
            margin = 0.0  # Can start with 0 and increase if needed
            exceedance = F.relu(pert_psd - masking_threshold + margin)
            
         
                        
            channel_loss = torch.mean(exceedance)
            total_loss += channel_loss
    return total_loss / (batch_size * channels)


def compute_stage2_loss(model, perturbed_audio, original_audio, target_sources, 
                        
                        alpha=0.05, window_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
    """"""
    Compute the Stage 2 loss for adversarial attack on Demucs.
    Balances separation error and psychoacoustic masking for imperceptibility.
    
    Args:
        model: Demucs v4 model instance
        perturbed_audio: Tensor of shape [batch, channels, samples] - the perturbed audio
        original_audio: Tensor of shape [batch, channels, samples] - the original audio
        target_sources: List of tensors - the ground truth sources
        alpha: Weight for psychoacoustic loss component
        window_size: Window size for STFT in psychoacoustic masking
        device: Device to use for computation
        
    Returns:
        total_loss: Combined loss balancing effectiveness and imperceptibility
        separation_loss: The effectiveness part of the loss
        psycho_loss: The imperceptibility part of the loss
    """"""
    # Compute separation loss (same as Stage 1)
    separation_loss = compute_stage1_loss(model, perturbed_audio, target_sources, device)
    
    # Compute psychoacoustic masking loss
    psycho_loss = compute_psychoacoustic_loss(
        perturbed_audio - original_audio,  # The perturbation
        original_audio,
        window_size=window_size,
        device=device
    )
    
    # Combine losses - separation_loss is already negated in compute_stage1_loss
    total_loss = separation_loss + alpha * psycho_loss
    
    return total_loss, separation_loss, psycho_loss

def run_stage2_attack(model, audio, sources, stage1_perturbation, output_dir, 
                     iterations=2000, epsilon=0.002, lr=0.0005, alpha_start=0.5,
                     alpha_max=1.0, window_size=2048, device='cuda'):
    """"""
    Run the Stage 2 attack (balance degrading separation with imperceptibility).
    
    Args:
        model: Demucs model instance
        audio: Original audio tensor
        sources: List of original source tensors
        stage1_perturbation: Best perturbation from stage 1
        output_dir: Directory to save results
        iterations: Number of attack iterations
        epsilon: L-infinity constraint
        lr: Learning rate
        alpha_start: Initial weight for psychoacoustic loss
        alpha_max: Maximum weight for psychoacoustic loss
        window_size: Window size for STFT in psychoacoustic masking
        device: Computation device
        
    Returns:
        torch.Tensor: Best perturbation found
    """"""
    print(f"Starting Stage 2 attack for {iterations} iterations...")
    # Create subdirectory for stage 2 results
    audio = audio.to(device).detach()

"""


def calculate_snr(audio, perturbation):
    """
    Calculate Signal-to-Noise Ratio in dB.
    
    Args:
        audio: Original audio signal (tensor)
        perturbation: Perturbation signal (tensor)
        
    Returns:
        float: SNR in dB
    """
    signal_power = torch.mean(audio ** 2)
    noise_power = torch.mean(perturbation ** 2)
    
    if noise_power.item() == 0:
        return float('inf')
        
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def plot_loss_over_iterations(losses, snrs, output_path, smoothing=10):
    """
    Create a plot showing the loss progression over iterations.
    
    Args:
        losses: List of loss values
        snrs: List of SNR values
        output_path: Path to save the plot
        smoothing: Window size for moving average smoothing
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot raw loss values
    iterations = range(1, len(losses) + 1)
    ax1.plot(iterations, losses, 'b-', alpha=0.3, label='Raw Loss')
    
    # Add smoothed version for clarity
    if smoothing > 0 and len(losses) > smoothing:
        # Apply moving average smoothing
        smoothed_losses = []
        for i in range(len(losses)):
            if i < smoothing:
                window = losses[:i+1]
            else:
                window = losses[i-smoothing+1:i+1]
            smoothed_losses.append(sum(window) / len(window))
        ax1.plot(iterations, smoothed_losses, 'r-', linewidth=2, label=f'Smoothed Loss (window={smoothing})')
    
    # Add loss annotations and styling
    ax1.set_title('Loss Over Iterations')
    ax1.set_ylabel('Loss Value (negative)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight best loss
    best_iter = np.argmin(losses)
    best_loss = losses[best_iter]
    ax1.plot(best_iter + 1, best_loss, 'go', markersize=8)
    ax1.annotate(f'Best: {best_loss:.4f}', 
                 xy=(best_iter + 1, best_loss),
                 xytext=(best_iter + 1 + len(losses)/20, best_loss),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                 fontsize=10)
    
    # Plot SNR on second subplot
    ax2.plot(iterations, snrs, 'g-', linewidth=2)
    ax2.set_title('Signal-to-Noise Ratio (SNR) Over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('SNR (dB)')
    ax2.grid(True, alpha=0.3)
    
    # Add SNR annotations
    final_snr = snrs[-1]
    ax2.annotate(f'Final SNR: {final_snr:.2f} dB', 
                 xy=(len(snrs), final_snr),
                 xytext=(len(snrs) - len(snrs)/5, final_snr + 0.5),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                 fontsize=10)
    
    # Improve layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    print(f"Loss visualization saved to {output_path}")


def run_stage1_attack(model, sources, output_dir, iterations=1000, epsilon=0.002, lr=0.01, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run the Stage 1 attack (focus on degrading separation without considering imperceptibility).
    Fixed to ensure device consistency. Added loss graph visualization.
    """
    print(f"Starting attack for {iterations} iterations...")
    print(f'iterations = {iterations}')
    print(f'epsilon: {epsilon}')
    print(f'lr: {lr}')
    print(f'device: {device}')
    
    # Extract and move audio to specified device
    audio = sources["mixture"].to(device)
    
   
    # Ensure all sources are on the same device
    device_sources = {}
    for key, tensor in sources.items():
        device_sources[key] = tensor.to(device)
    
    # Detach the audio to ensure we don't compute gradients through it
    audio = audio.detach()
    
    # Create perturbation with gradients enabled on the same device
    perturbation = torch.zeros_like(audio, device=device)
    perturbation.requires_grad_()
    
 
    # Create optimizer
    optimizer = torch.optim.Adam([perturbation], lr=lr)
    
    # Track metrics
    losses = []
    snrs = []
    best_loss = float('inf')
    best_perturbation = None
    
    # Attack loop
    for i in tqdm(range(iterations)):
        # Zero gradients
        optimizer.zero_grad()
        
        # Clamp perturbation directly
        perturbation.data = torch.clamp(perturbation, -epsilon, epsilon)

        # Create perturbed audio
        perturbed_audio = audio + perturbation

        # Compute loss - all tensors should be on the same device now
        loss = compute_stage1_loss(model, perturbed_audio, device_sources, device)
        
        # Backpropagate
        loss.backward()
        
        # Update perturbation
        optimizer.step()
        
        # Clamp again after update
        perturbation.data = torch.clamp(perturbation, -epsilon, epsilon)
    
        # Calculate metrics
        with torch.no_grad():
            # Compute SNR
            snr = calculate_snr(audio, perturbation)
            
            # Track progress
            losses.append(loss.item())
            snrs.append(snr)
            
            # Save best perturbation
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_perturbation = perturbation.clone().detach()
                
                print(f"Iteration {i+1}/{iterations}, New best loss: {loss.item():.4f}, SNR: {snr:.2f} dB")
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{iterations}, Loss: {loss.item():.4f}, SNR: {snr:.2f} dB")
    
    # Create and save loss visualization
    comparisons_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)
    loss_plot_path = os.path.join(comparisons_dir, "loss_progression.png")
    plot_loss_over_iterations(losses, snrs, loss_plot_path)
    
    # Return the best perturbation found - make sure it's detached
    return best_perturbation.detach()



    stage2_dir = os.path.join(output_dir, "stage2")
    os.makedirs(stage2_dir, exist_ok=True)

    # Initialize perturbation with stage 1 result
    perturbation = stage1_perturbation.clone().detach().requires_grad_(True)
    
    # Create optimizer
    optimizer = torch.optim.Adam([perturbation], lr=lr)
    
    # Track metrics
    losses = []
    sep_losses = []
    psyco_losses = []
    snrs = []
    alphas = []
    best_loss = float('inf')
    best_perturbation = None
    alpha = alpha_start
    
    # Attack loop
    for i in tqdm(range(iterations)):
        # Zero gradients
        optimizer.zero_grad()
        """
        # Apply perturbation with L-infinity constraint
        with torch.no_grad():
            delta = torch.clamp(perturbation, -epsilon, epsilon)
            perturbation.data.copy_(delta)
        """
        # Create perturbed audio
        perturbed_audio = audio + perturbation
        
        # Compute combined loss for stage 2
        loss, sep_loss, psyco_loss = compute_stage2_loss(
            model, perturbed_audio, audio, sources, 
            alpha=alpha, window_size=window_size, device=device
        )
        # Backpropagate
        loss.backward()
        
        # Update perturbation
        optimizer.step()
        
        
        
        # Apply L-infinity constraint again after optimization step
        with torch.no_grad():
            delta = torch.clamp(perturbation, -epsilon, epsilon)
            perturbation.data.copy_(delta)
        #Calculate metrics
        with torch.no_grad():
            # Compute SNR
            snr = calculate_snr(audio, perturbation)
            
            # Track progress
            losses.append(loss.item())
            sep_losses.append(sep_loss.item())
            psyco_losses.append(psyco_loss.item())
            snrs.append(snr)
            alphas.append(alpha)
            
            # Save best perturbation (prioritizing imperceptibility)
            if i > 0 and loss.item() < best_loss:
                best_loss = loss.item()
         
                best_perturbation = perturbation.clone().detach()
                
                # Save intermediate best
                final_perturbed = audio + best_perturbation
                best_path = os.path.join(stage2_dir, "best_perturbed.wav")
                torchaudio.save(best_path, final_perturbed.cpu(), 44100)
                
                print(f"Iteration {i+1}/{iterations}, New best! Sep: {sep_loss.item():.4f}, "
                      f"Psyco: {psyco_loss.item():.4f}, SNR: {snr:.2f} dB, Alpha: {alpha:.4f}")
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{iterations}, Sep: {sep_loss.item():.4f}, "
                      f"Psyco: {psyco_loss.item():.4f}, SNR: {snr:.2f} dB, Alpha: {alpha:.4f}")
        
        # Dynamically adjust alpha - increase if separation is good, decrease if it's poor
        if i % 50 == 0 and i > 0:
            if sep_loss.item() > 0.9 * max(sep_losses[-50:]):
                # Separation is still good, increase alpha to focus on imperceptibility
                alpha = min(alpha * 1.2, alpha_max)
            else:
                # Separation is degrading, decrease alpha to focus on attack effectiveness
                alpha = max(alpha * 0.8, alpha_start * 0.1)
        
        # Save intermediate results occasionally
        if (i + 1) % 200 == 0 or i == iterations - 1:
            # Create perturbed audio
            final_perturbed = audio + perturbation
            
            # Save perturbed audio
            perturbed_path = os.path.join(stage2_dir, f"perturbed_iter_{i+1}.wav")
            torchaudio.save(perturbed_path, final_perturbed.cpu().detach(), 44100)
            
            # Save perturbation
            perturbation_path = os.path.join(stage2_dir, f"perturbation_iter_{i+1}.wav")
            torchaudio.save(perturbation_path, perturbation.cpu().detach(), 44100)
    
    # Plot metrics over iterations
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(sep_losses)
    plt.title('Separation Loss vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Separation Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(psyco_losses)
    plt.title('Psychoacoustic Loss vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Psychoacoustic Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(snrs)
    plt.title('SNR vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('SNR (dB)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(alphas)
    plt.title('Alpha vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stage2_dir, 'stage2_metrics.png'))
    plt.close()
    

        
    # Return the best perturbation found
    return best_perturbation




def visualize_audio_differences(original_audio, perturbed_audio, 
                              channel=0, sample_rate=44100,
                              save_path="wave_forms.png"):
    """
    Visualize differences between highly similar audio waveforms.
    """
    # Convert tensors to numpy
    if isinstance(original_audio, torch.Tensor):
        original_audio = original_audio.cpu().detach().numpy()
    if isinstance(perturbed_audio, torch.Tensor):
        perturbed_audio = perturbed_audio.cpu().detach().numpy()
    
    # Get single channel data
    original = original_audio[channel]
    perturbed = perturbed_audio[channel]
    

    
    # Create time axis
    duration = len(original) / sample_rate
    time = np.linspace(0, duration, len(original))
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Overlay the waveforms
    axes[0].plot(time, perturbed, color='#e74c3c', label='Perturbed', linewidth=1)
    axes[0].plot(time, original, color='#3498db', label='Original', linewidth=1)
    axes[0].set_title('Waveform Comparison (Overlaid)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Show with vertical offset
    offset = 0.1 * max(np.max(np.abs(original)), np.max(np.abs(perturbed)))
    axes[1].plot(time, perturbed, color='#e74c3c', label='Perturbed', linewidth=1)
    axes[1].plot(time, original - offset, color='#3498db', label='Original (offset)', linewidth=1)
    axes[1].set_title('Waveform Comparison (With Offset)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    


    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")
    


def process_full_song(sources, output_dir, model, song_name, device="cuda", lr=0.01, epsilon=0.002, iterations=1000):
    """
    Process a full song using your existing attack function with pre-loaded sources.
    
    Args:
        sources: Dictionary of source tensors (already loaded)
        output_dir: Directory to save results
        model: Your Demucs model
        song_name: Name of the song to create a subdirectory for results
        device: Device to use for processing
        lr: Learning rate for the attack
        epsilon: Epsilon constraint for the attack
        iterations: Number of iterations for the attack
        
    Returns:
        Tuple of (processed_audio, perturbation, reassembled_original)
    """
    # Create a wrapper for your attack function
    def attack_wrapper(segment_sources):
        # Adapt your existing attack function to work with segments
        perturbed_audio = run_stage1_attack(
            model=model,
            sources=segment_sources,
            output_dir=output_dir,
            iterations=iterations,
            epsilon=epsilon,
            lr=lr,
            device=device
        )
        return perturbed_audio

    # Modified version of process_audio_in_segments that works with tensors instead of files
    processed_audio, perturbation, reassembled_original = process_audio_segments(
        sources=sources,
        output_dir=output_dir,
        attack_function=attack_wrapper,
        song_name=song_name,  # Pass the song name to create a subdirectory
        segment_length=5.0,
        sample_rate=44100,
        overlap=0.1,
        device=device
    )
    
    return processed_audio, perturbation, reassembled_original

def process_audio_segments(
        sources,
        output_dir,
        attack_function, 
        song_name,
        segment_length=5.0, 
        sample_rate=44100, 
        overlap=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Process audio by breaking the main audio (in sources["mixture"]) into segments,
    applying an attack to each segment using corresponding segments from sources,
    then concatenating the results. 
    Fixed to ensure device consistency.
    """
    # Create song-specific directory inside output_dir
    song_dir = os.path.join(output_dir, song_name)
    os.makedirs(song_dir, exist_ok=True)
    
    # Create comparisons directory for visualizations
    comparisons_dir = os.path.join(song_dir, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Define output file paths in the song-specific directory
    perturbed_audio_path = os.path.join(song_dir, "perturbed_mixture.wav")
    perturbation_path = os.path.join(song_dir, "final_perturbation.wav")
    
    # Get main audio details from sources["mixture"]
    # Make sure it's on CPU for initialization step
    mixture = sources["mixture"].cpu()
    channels, num_samples = mixture.shape
    duration = num_samples / sample_rate
    print(f"Mixture duration: {duration:.2f} seconds ({channels} channels)")
    
    # Calculate segment samples and overlap samples
    segment_samples = int(segment_length * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step_samples = segment_samples - overlap_samples
    
    # Calculate number of segments
    num_segments = max(1, int(np.ceil((num_samples - overlap_samples) / step_samples)))
    print(f"Processing {num_segments} segments of {segment_length} seconds each")
    
    # Prepare output tensors on CPU for accumulating results
    processed_mixture = torch.zeros_like(mixture, device='cpu')
    total_perturbation = torch.zeros_like(mixture, device='cpu')
    reassembled_original = torch.zeros_like(mixture, device='cpu')
    
    # Process each segment
    for i in tqdm(range(num_segments)):
        start_sample = i * step_samples
        end_sample = min(start_sample + segment_samples, num_samples)
        actual_seg_length = end_sample - start_sample
        
        print(f"Processing segment {i+1}/{num_segments} (samples {start_sample}-{end_sample})")
        
        # Build a dictionary for current segment for each source - all on CPU initially
        current_segments = {}
        for key, tensor in sources.items():
            # Ensure source tensor is on CPU for this operation
            cpu_tensor = tensor.cpu()
            
            if actual_seg_length < segment_samples:
                # Pad with zeros if the segment is shorter than expected
                seg = torch.zeros((channels, segment_samples), dtype=cpu_tensor.dtype, device='cpu')
                seg[:, :actual_seg_length] = cpu_tensor[:, start_sample:end_sample]
            else:
                seg = cpu_tensor[:, start_sample:end_sample].clone()
            current_segments[key] = seg
        
        # Store original mixture segment for verification (only actual length)
        if actual_seg_length < segment_samples:
            reassembled_original[:, start_sample:end_sample] = current_segments["mixture"][:, :actual_seg_length]
        else:
            reassembled_original[:, start_sample:end_sample] = current_segments["mixture"]
        
        # Move each segment to target device for processing - do this as a separate step
        device_segments = {}
        for key, tensor in current_segments.items():
            device_segments[key] = tensor.to(device)
        
        # Save original segment on CPU
        original_segment = current_segments["mixture"]
        
        # Apply attack to the segment
        try:
            # The attack_function should handle device consistency internally
            segment_perturbation = attack_function(device_segments)
            
            # Move perturbation back to CPU immediately
            segment_perturbation = segment_perturbation.to('cpu')
            
            # Calculate perturbed segment
            processed_segment = original_segment + segment_perturbation
                
            # Store in output tensors (already on CPU)
            if actual_seg_length < segment_samples:
                processed_mixture[:, start_sample:end_sample] = processed_segment[:, :actual_seg_length]
                total_perturbation[:, start_sample:end_sample] = segment_perturbation[:, :actual_seg_length]
            else:
                processed_mixture[:, start_sample:end_sample] = processed_segment
                total_perturbation[:, start_sample:end_sample] = segment_perturbation
                
        except Exception as e:
            print(f"Error processing segment {i+1}: {e}")
            # In case of error, print more detailed information
            print(f"Tensor device information for debugging:")
            for key, tensor in device_segments.items():
                print(f"  {key}: {tensor.device}")
            
            print("Traceback:")
            import traceback
            traceback.print_exc()
            
            # Use original segment in case of error
            processed_mixture[:, start_sample:end_sample] = original_segment[:, :actual_seg_length] if actual_seg_length < segment_samples else original_segment
            # Continue with next segment
            continue
    
    # Make sure tensors are on CPU before saving
    processed_mixture = processed_mixture.cpu()
    total_perturbation = total_perturbation.cpu()
    
    # Save output audio files for the main mixture
    print(f"Saving perturbed mixture to {perturbed_audio_path}")
    torchaudio.save(perturbed_audio_path, processed_mixture, sample_rate)
   
    print(f"Saving perturbation to {perturbation_path}")
    torchaudio.save(perturbation_path, total_perturbation, sample_rate)
   
    # Save waveform visualization in comparisons folder
    visualize_audio_differences(
        original_audio=mixture.cpu(),
        perturbed_audio=processed_mixture,
        sample_rate=sample_rate,
        save_path=os.path.join(comparisons_dir, "wave_forms.png")
    )
   
    # Calculate and print statistics (for the main mixture)
   

    return processed_mixture, total_perturbation, reassembled_original

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='White-box adversarial attack on Demucs v4')
    parser.add_argument('--input_file', type=str, required=True, help='Directory containing mixture and source files')

    parser.add_argument('--output_dir', type=str, default='./attack_results', help='Directory to save results')
    parser.add_argument('--model', type=str, default="htdemucs", help='Demucs model name')
    parser.add_argument('--stage1_iter', type=int, default=3000, help='Number of Stage 1 iterations')
    #parser.add_argument('--stage2_iter', type=int, default=2000, help='Number of Stage 2 iterations')
    parser.add_argument('--epsilon', type=float, default=0.0035, help='L-infinity constraint')
    parser.add_argument('--stage1_lr', type=float, default=0.00005, help='Learning rate for Stage 1')
   # parser.add_argument('--stage2_lr', type=float, default=0.005, help='Learning rate for Stage 2')
  #  parser.add_argument('--alpha', type=float, default=0.05, help='Initial weight for psychoacoustic loss')
  #  parser.add_argument('--alpha_max', type=float, default=1.0, help='Maximum weight for psychoacoustic loss')
 #  parser.add_argument('--window_size', type=int, default=2048, help='Window size for STFT')
  #  parser.add_argument('--skip_stage1', action='store_true', help='Skip Stage 1 and load from stage1_result.pt')
  #  parser.add_argument('--skip_stage2', action='store_true', help='Skip Stage 2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--save_sources', action='store_true', help='Save separated sources to output directory')
    return parser.parse_args()

def load_and_separate_audio(mixture_path, model, output_dir=None, save_sources=False, 
                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a mixture audio file and separate it into sources using Demucs.
    
    Args:
        mixture_path: Path to the mixture audio file
        model: Demucs model instance
        output_dir: Directory to save separated sources if save_sources is True
        save_sources: Whether to save the separated sources
        device: Device to use for computation
        
    Returns:
        dict: Dictionary with source names as keys and audio tensors as values
        int: Sample rate
    """
    # Load the mixture audio
    mixture, sample_rate = torchaudio.load(mixture_path)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Move mixture to device
    mixture = mixture.to(device)
    
    # Ensure mixture has batch dimension [batch, channels, samples]
    if mixture.dim() == 2:
        mixture_batch = mixture.unsqueeze(0)
    else:
        mixture_batch = mixture
    
    print(f"Separating sources using {model.__class__.__name__}...")
    
    # Separate using Demucs
    with torch.no_grad():
        sources = apply_model(model, mixture_batch, device=device)
    
    # Remove batch dimension
    sources = sources.squeeze(0)  # Now [sources, channels, samples]
    
    # Create dictionary with sources
    source_names = ['drums', 'bass', 'other', 'vocals']
    audio_tensors = {}
    
    # Create output directory if needed
    if save_sources and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    for i, name in enumerate(source_names):
        audio_tensors[name] = sources[i]
        
        # Save sources if requested
        if save_sources and output_dir is not None:
            source_path = os.path.join(output_dir, f"{name}.wav")
            # Make sure the tensor is on CPU before saving
            torchaudio.save(source_path, sources[i].cpu(), sample_rate)
            print(f"Saved {name} source to {source_path}")
    
    # Add the original mixture
    audio_tensors['mixture'] = mixture
    
    # Save mixture if requested
    if save_sources and output_dir is not None:
        mixture_out_path = os.path.join(output_dir, "mixture.wav")
        # Make sure the tensor is on CPU before saving
        torchaudio.save(mixture_out_path, mixture.cpu(), sample_rate)
        print(f"Saved mixture to {mixture_out_path}")
    
    return audio_tensors, sample_rate
    
def main():
    """
    Main function to run the adversarial attack with fixed saving logic.
    Updated to ensure comparisons directory exists before attack.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Extract song name from the input file path
    song_name = os.path.splitext(os.path.basename(args.input_file))[0]
    print(f"Processing song: {song_name}")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create song-specific output directory
    song_output_dir = os.path.join(args.output_dir, song_name)
    os.makedirs(song_output_dir, exist_ok=True)
    
    # Create subdirectories
    originals_dir = os.path.join(song_output_dir, "originals")
    estimated_dir = os.path.join(song_output_dir, "estimated")
    comparisons_dir = os.path.join(song_output_dir, "comparisons")
    
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(estimated_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Set device and print info
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Attack parameters: epsilon={args.epsilon}, lr={args.stage1_lr}, iterations={args.stage1_iter}")
    
    try:
        # Load Demucs model
        print(f"Loading Demucs model: {args.model}")
        model = get_model(args.model)
        model.to(device)
        
        # Load mixture and separate into sources
        print(f"Loading mixture from {args.input_file}")
        original_mixture, sample_rate = torchaudio.load(args.input_file)
        
        # Save original mixture
        original_mixture_path = os.path.join(originals_dir, "mixture.wav")
        torchaudio.save(original_mixture_path, original_mixture, sample_rate)
        print(f"Saved original mixture to {original_mixture_path}")
        
        # Separate original sources
        print("Separating original sources...")
        original_mixture_device = original_mixture.to(device)
        if original_mixture_device.dim() == 2:
            original_mixture_batch = original_mixture_device.unsqueeze(0)
        else:
            original_mixture_batch = original_mixture_device
        
        with torch.no_grad():
            original_sources_batch = apply_model(model, original_mixture_batch, device=device)
        
        original_sources_batch = original_sources_batch.squeeze(0)  # [sources, channels, samples]
        
        # Create sources dictionary
        source_names = ['drums', 'bass', 'other', 'vocals']
        sources = {'mixture': original_mixture}
        
        # Save original separated sources
        for i, name in enumerate(source_names):
            sources[name] = original_sources_batch[i].cpu()
            source_path = os.path.join(originals_dir, f"{name}.wav")
            torchaudio.save(source_path, sources[name], sample_rate)
            print(f"Saved original {name} to {source_path}")
        
        # Process the audio by segments
        print(f"Starting adversarial attack on {song_name}")
        processed_audio, perturbation, reassembled = process_audio_segments(
            sources=sources,
            output_dir=args.output_dir,
            attack_function=lambda segment_sources: run_stage1_attack(
                model=model,
                sources=segment_sources,
                output_dir=song_output_dir,  # Pass song_output_dir to run_stage1_attack
                iterations=args.stage1_iter,
                epsilon=args.epsilon,
                lr=args.stage1_lr,
                device=device
            ),
            song_name=song_name,
            segment_length=5.1,
            sample_rate=sample_rate,
            overlap=0.1,
            device=device
        )
                    
        # Ensure perturbed mixture is saved correctly
        perturbed_mixture_path = os.path.join(song_output_dir, "perturbed_mixture.wav")
        print(f"Saving perturbed mixture to {perturbed_mixture_path}")
        perturbed_mixture = processed_audio.cpu()
        torchaudio.save(perturbed_mixture_path, perturbed_mixture, sample_rate)
        

        # Separate sources from the perturbed audio
        print("Separating sources from perturbed audio...")
        perturbed_mixture_device = perturbed_mixture.to(device)
        if perturbed_mixture_device.dim() == 2:
            perturbed_mixture_batch = perturbed_mixture_device.unsqueeze(0)
        else:
            perturbed_mixture_batch = perturbed_mixture_device
        
        with torch.no_grad():
            perturbed_sources_batch = apply_model(model, perturbed_mixture_batch, device=device)
        
        perturbed_sources_batch = perturbed_sources_batch.squeeze(0)  # [sources, channels, samples]
        
        # Save perturbed sources
        for i, name in enumerate(source_names):
            source_path = os.path.join(estimated_dir, f"{name}.wav")
            perturbed_source = perturbed_sources_batch[i].cpu()
            torchaudio.save(source_path, perturbed_source, sample_rate)
            print(f"Saved perturbed {name} to {source_path}")
            
            # Calculate difference with original source
            source_diff = torch.mean(torch.abs(sources[name] - perturbed_source)).item()
            print(f"Average difference between original and perturbed {name}: {source_diff}")
        
        # Save perturbed mixture in estimated folder for completeness
        torchaudio.save(os.path.join(estimated_dir, "mixture.wav"), perturbed_mixture, sample_rate)
        
        # Create waveform visualization comparing original sources and perturbed sources
        for i, name in enumerate(source_names):
            compare_path = os.path.join(comparisons_dir, f"{name}_comparison.png")
            original_source = sources[name].numpy()
            perturbed_source = perturbed_sources_batch[i].cpu().numpy()
            
            try:
                visualize_audio_differences(
                    original_audio=original_source,
                    perturbed_audio=perturbed_source,
                    sample_rate=sample_rate,
                    save_path=compare_path
                )
                print(f"Created visualization comparing original and perturbed {name}")
            except Exception as e:
                print(f"Error creating visualization for {name}: {e}")
        
        # Calculate attack success metrics
        stats_file = os.path.join(song_output_dir, "separation_metrics.txt")
        with open(stats_file, "w") as f:
            f.write(f"Attack Success Metrics for {song_name}\n")
            f.write("====================================\n\n")
            
            # For each source, compute similarity between original separation and perturbed separation
            for i, name in enumerate(source_names):
                original_source = sources[name]
                perturbed_source = perturbed_sources_batch[i].cpu()
                
                # Compute correlation
                corr = torch.mean(
                    torch.nn.functional.cosine_similarity(
                        original_source.flatten().unsqueeze(0),
                        perturbed_source.flatten().unsqueeze(0)
                    )
                ).item()
                
                # Compute L1 difference
                l1_diff = torch.mean(torch.abs(original_source - perturbed_source)).item()
                
                # Write to file
                f.write(f"Source: {name}\n")
                f.write(f"  Cosine similarity: {corr:.4f}\n")
                f.write(f"  L1 difference: {l1_diff:.6f}\n\n")
            
            f.write("Lower similarity and higher difference indicate more successful attack.\n")
        
        print(f"Attack on '{song_name}' completed successfully!")
        print(f"Results saved in: {song_output_dir}")
        print(f"  Original sources: {originals_dir}")
        print(f"  Estimated sources (after attack): {estimated_dir}")
        print(f"  Visualizations: {comparisons_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)