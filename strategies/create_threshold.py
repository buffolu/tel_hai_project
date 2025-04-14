# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:33:42 2025

@author: igor
"""


import numpy as np
import librosa
from scipy import signal


class PsychoacousticMasker:
    """
    Implements psychoacoustic masking threshold computation.
    Adapted from the paper's implementation.
    """
    def __init__(self, window_size=2048, sample_rate=44100):
        self.window_size = window_size
        self.sample_rate = sample_rate
        
    def compute_PSD_matrix(self, audio):
        """
        Compute the Power Spectral Density matrix.
        First, perform STFT, then compute PSD, and finally normalize PSD.
        """
        # Use librosa for STFT - modified to match original code
        win = np.sqrt(8.0/3.) * librosa.core.stft(audio, center=False)
        
        # Compute PSD and normalize
        z = abs(win / self.window_size)
        psd_max = np.max(z*z)
        psd = 10 * np.log10(z * z + 0.0000000000000000001)  # Match the original small constant
        PSD = 96 - np.max(psd) + psd
    
        return PSD, psd_max
    
    def bark(self, f):
        """
        Convert frequency to Bark scale.
        Returns the bark-scale value for input frequency f (in Hz).
        """
        return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan(pow(f/7500.0, 2))
    
    def quiet(self, f):
        """
        Returns threshold in quiet measured in SPL at frequency f with an offset.
        """
        thresh = 3.64 * pow(f*0.001, -0.8) - 6.5 * np.exp(-0.6 * pow(0.001*f-3.3, 2)) + 0.001 * pow(0.001*f, 4) - 12
        return thresh
    
    def two_slopes(self, bark_psd, delta_TM, bark_maskee):
        """
        Returns the masking threshold for each masker using two slopes as the spread function.
        """
        Ts = []
        for tone_mask in range(bark_psd.shape[0]):
            bark_masker = bark_psd[tone_mask, 0]
            dz = bark_maskee - bark_masker
            zero_index = np.argmax(dz > 0)
            sf = np.zeros(len(dz))
            sf[:zero_index] = 27 * dz[:zero_index]
            sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
            T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
            Ts.append(T)
        return Ts
    
    def compute_th(self, PSD, barks, ATH, freqs):
        """
        Returns the global masking threshold.
        """
        # Identification of tonal maskers
        length = len(PSD)
        masker_index = signal.argrelextrema(PSD, np.greater)[0]
        
        # Delete the boundary of maskers for smoothing
        if 0 in masker_index:
            masker_index = np.delete(masker_index, np.where(masker_index == 0)[0])
        if length - 1 in masker_index:
            masker_index = np.delete(masker_index, np.where(masker_index == length - 1)[0])
            
        num_local_max = len(masker_index)
        
        if num_local_max == 0:
            # Return ATH if no maskers found
            return ATH
        
        # Treat all the maskers as tonal (conservative approach)
        # Smooth the PSD
        p_k = pow(10, PSD[masker_index]/10.)    
        p_k_prev = pow(10, PSD[masker_index - 1]/10.)
        p_k_post = pow(10, PSD[masker_index + 1]/10.)
        P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)
        
        # Bark_psd: [bark, P_TM, index]
        _BARK = 0
        _PSD = 1
        _INDEX = 2
        bark_psd = np.zeros([num_local_max, 3])
        bark_psd[:, _BARK] = barks[masker_index]
        bark_psd[:, _PSD] = P_TM
        bark_psd[:, _INDEX] = masker_index
        
        # Delete maskers that don't have the highest PSD within 0.5 Bark
        i = 0
        while i < bark_psd.shape[0]:
            if i + 1 >= bark_psd.shape[0]:
                break
                
            next_i = i + 1
            while next_i < bark_psd.shape[0] and bark_psd[next_i, _BARK] - bark_psd[i, _BARK] < 0.5:
                # Check if masker is above quiet threshold
                if self.quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                    bark_psd = np.delete(bark_psd, i, axis=0)
                    if i >= bark_psd.shape[0]:
                        break
                    continue
                    
                if next_i >= bark_psd.shape[0]:
                    break
                    
                if bark_psd[i, _PSD] < bark_psd[next_i, _PSD]:
                    bark_psd = np.delete(bark_psd, i, axis=0)
                    if i >= bark_psd.shape[0]:
                        break
                    continue
                else:
                    bark_psd = np.delete(bark_psd, next_i, axis=0)
                    continue
                
            i += 1
        
        if bark_psd.shape[0] == 0:
            # Return ATH if no maskers remain after filtering
            return ATH
            
        # Compute the individual masking threshold
        delta_TM = 1 * (-6.025 - 0.275 * bark_psd[:, 0])
        Ts = self.two_slopes(bark_psd, delta_TM, barks)
        Ts = np.array(Ts)
        
        # Compute the global masking threshold
        theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.)
        theta_x = 10 * np.log10(theta_x)
        
        return theta_x
    
    def generate_th(self, audio):
        """
        Returns the masking thresholds and the max PSD of the audio.
        
        Args:
            audio: Audio signal (mono)
            
        Returns:
            theta_xs: Masking thresholds
            psd_max: Maximum PSD value
            freqs: Frequencies
        """
        # Get the PSD matrix
        PSD, psd_max = self.compute_PSD_matrix(audio)
        
        # Get the frequencies and convert to Bark scale
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.window_size)
        barks = self.bark(freqs)
        
        # Compute the quiet threshold
        ATH = np.zeros(len(barks)) - np.inf
        bark_ind = np.argmax(barks > 1)
        ATH[bark_ind:] = self.quiet(freqs[bark_ind:])
        
        # Compute the global masking threshold for each window
        theta_xs = []
        for i in range(PSD.shape[1]):
            theta_xs.append(self.compute_th(PSD[:, i], barks, ATH, freqs))
            
        theta_xs = np.array(theta_xs)
        return theta_xs, psd_max, freqs


class FrequencyDomainTransform:
    """
    Transforms audio to the frequency domain for psychoacoustic masking.
    """
    def __init__(self, window_size=2048, sample_rate=44100):
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.hop_length = window_size // 4
        
    def __call__(self, audio, psd_max_ori=None):
        """
        Transform audio to frequency domain and normalize.
        
        Args:
            audio: Audio signal (mono)
            psd_max_ori: Maximum PSD value from original audio
            
        Returns:
            PSD: Power spectral density
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.window_size, 
                          hop_length=self.hop_length, 
                          window='hann', center=False)
        
        # Compute PSD
        z = abs(stft / self.window_size)
        
        if psd_max_ori is None:
            psd_max = np.max(z*z)
        else:
            psd_max = psd_max_ori
            
        psd = 10 * np.log10(z * z + 1e-20)
        PSD = 96 - np.max(psd) + psd
        
        return PSD
