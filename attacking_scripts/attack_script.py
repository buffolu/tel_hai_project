import numpy as np
import os
import torch
import torch.nn.functional as F
import torchaudio
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sys

def compute_stage1_loss(model, perturbed_audio, target_sources, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute loss for stage 1 attack - focus on degrading separation quality.
    
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

def run_adversarial_attack(model, sources, output_dir, iterations=1000, epsilon=0.00035, lr=0.01, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run adversarial attack on the full song to degrade separation quality.
    
    Args:
        model: Demucs model
        sources: Dictionary of source tensors
        output_dir: Directory to save results
        iterations: Number of optimization iterations
        epsilon: L-infinity constraint for the perturbation
        lr: Learning rate for optimization
        device: Computation device
        
    Returns:
        torch.Tensor: Optimized perturbation
    """
    print(f"Starting attack for {iterations} iterations...")
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='White-box adversarial attack on Demucs v4')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output_dir', type=str, default='./attack_results', help='Directory to save results')
    parser.add_argument('--model', type=str, default="htdemucs_ft", help='Demucs model name')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of attack iterations')
    parser.add_argument('--epsilon', type=float, default=0.0005, help='L-infinity constraint')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate for optimization')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--save_sources', action='store_true', help='Save separated sources to output directory')
    return parser.parse_args()

def main():
    """
    Main function to run the adversarial attack on a whole song.
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
    originals_dir = os.path.join(song_output_dir, "separation_prior_attack")
    estimated_dir = os.path.join(song_output_dir, "separation_after_attack")
    comparisons_dir = os.path.join(song_output_dir, "comparisons")
    
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(estimated_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Set device and print info
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Attack parameters: epsilon={args.epsilon}, lr={args.lr}, iterations={args.iterations}")
    
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
        
        # Apply adversarial attack to the whole song
        print(f"Starting adversarial attack on {song_name}")
        perturbation = run_adversarial_attack(
            model=model,
            sources=sources,
            output_dir=song_output_dir,
            iterations=args.iterations,
            epsilon=args.epsilon,
            lr=args.lr,
            device=device
        )
        
        # Create perturbed mixture
        perturbed_mixture = original_mixture + perturbation.cpu()
        
        # Save perturbed mixture
        perturbed_mixture_path = os.path.join(song_output_dir, "perturbed_mixture.wav")
        torchaudio.save(perturbed_mixture_path, perturbed_mixture, sample_rate)
        print(f"Saved perturbed mixture to {perturbed_mixture_path}")
        
        # Save perturbation alone
        perturbation_path = os.path.join(song_output_dir, "final_perturbation.wav")
        torchaudio.save(perturbation_path, perturbation.cpu(), sample_rate)
        print(f"Saved perturbation to {perturbation_path}")
        
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
        
        # Create main waveform comparison
        main_compare_path = os.path.join(comparisons_dir, "wave_forms.png")
        visualize_audio_differences(
            original_audio=original_mixture,
            perturbed_audio=perturbed_mixture,
            sample_rate=sample_rate,
            save_path=main_compare_path
        )
        print("Created visualization comparing original and perturbed mixtures")
        
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
            
            # Calculate SNR of the perturbation
            snr = calculate_snr(original_mixture, perturbation.cpu())
            f.write(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB\n\n")
            
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