# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:44:41 2025

@author: igor
"""
import os
import librosa
import numpy as np
import mir_eval
def compute_separation_metrics(original_dir, adversarial_dir, target_stems=None):
    """
    Compute separation quality metrics between original and adversarial stems.
    
    Args:
        original_dir: Directory containing original separated stems
        adversarial_dir: Directory containing adversarially separated stems
        target_stems: List of stems to compute metrics for (default: all)
        
    Returns:
        dict: Dictionary of separation metrics
    """
    default_stems = ["drums", "bass", "vocals", "other"]
    stems = target_stems or default_stems
    
    metrics = {}
    all_sdrs = []
    all_sirs = []
    all_sars = []
    
    # Load all stems and compute metrics
    for stem in stems:
        orig_path = os.path.join(original_dir, f"{stem}.wav")
        adv_path = os.path.join(adversarial_dir, f"{stem}.wav")
        
        if not os.path.exists(orig_path) or not os.path.exists(adv_path):
            continue
            
        # Load audio
        orig_audio, sr = librosa.load(orig_path, sr=None, mono=True)
        adv_audio, _ = librosa.load(adv_path, sr=sr, mono=True)
        
        # Make sure lengths match
        min_len = min(len(orig_audio), len(adv_audio))
        orig_audio = orig_audio[:min_len]
        adv_audio = adv_audio[:min_len]
        
        # Compute BSS metrics
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            orig_audio.reshape(1, -1), adv_audio.reshape(1, -1)
        )
        
        metrics[stem] = {
            'sdr': sdr[0],
            'sir': sir[0],
            'sar': sar[0]
        }
        
        all_sdrs.append(sdr[0])
        all_sirs.append(sir[0])
        all_sars.append(sar[0])
    
    # Calculate average metrics
    if all_sdrs:
        metrics['avg'] = {
            'sdr': np.mean(all_sdrs),
            'sir': np.mean(all_sirs),
            'sar': np.mean(all_sars)
        }
        
        # Calculate combined score (lower is better for our adversarial goal)
        weights = {'sdr': 0.6, 'sir': 0.3, 'sar': 0.1}
        combined_score = (
            weights['sdr'] * metrics['avg']['sdr'] +
            weights['sir'] * metrics['avg']['sir'] +
            weights['sar'] * metrics['avg']['sar']
        )
        metrics['combined_score'] = combined_score
    else:
        metrics['avg'] = {'sdr': 0, 'sir': 0, 'sar': 0}
        metrics['combined_score'] = 0
    
    return metrics
