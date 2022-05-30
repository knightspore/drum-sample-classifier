import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchaudio import transforms
import numpy as np

# AudioUtil Processes Audio Files fed into the DataLoader

class AudioUtil():
    # --------------------------------------------------
    # Read Audio
    # --------------------------------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    # --------------------------------------------------
    # Convert the audio to desired number of channels
    # --------------------------------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        
        if (sig.shape[0] == new_channel):
            return aud # Nothing to do

        if (new_channel == 1): 
            # Convert to mono by selecting only channel one
            resig = sig[:1, :]
        else:
            # Convert to setereo by duplicating channel one'
            resig = torch.cat([sig, sig])
        
        return ((resig, sr))

    # --------------------------------------------------
    # Resample one channel at a time
    # --------------------------------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            return aud
        
        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1): 
            # Resample second channel and merge both
            retwo = torchaudio.transforms.Resample(sr,newsr)(sig[1:,:])
            resig = torch.cat([resig,retwo])

        return ((resig, newsr))

    # --------------------------------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # --------------------------------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate
            sig = sig[:,:max_len]
        
        elif (sig_len < max_len):
            # Pad signal end
            pad_len = max_len - sig_len
            pad_end = torch.zeros((num_rows, pad_len))
            sig = torch.cat((sig, pad_end), 1)

        return ((sig, sr))

    # --------------------------------------------------
    # Generate a spectrogram
    # --------------------------------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 120

        # Spec has shape [channel, n_mels, time] where channel is mono, stereo, etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to dB
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    
    # --------------------------------------------------
    # Augment Spectrogram
    # --------------------------------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return (aug_spec)
