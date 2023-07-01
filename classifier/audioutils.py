import torch
import torchaudio 
from torchaudio import transforms

class AudioUtils():

    '''
    TODO: Add DATA Augmentation methods: Time shift, time & frequency masking
    '''

    '''
    Load an audio file and return signal tensor (ch, signal_len) & sample rate
    '''
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    '''
    Convert the audio to mono
    '''
    @staticmethod
    def convert_mono(audio):
        if audio.shape[0] == 1:
            return audio
        
        return audio[:1, :]
    
    '''
    Resize the audio to max_samples by truncating or padding
    '''
    @staticmethod
    def resize(audio, max_samples):
        signal_len = audio.shape[1]

        if signal_len == max_samples:
            return audio
        
        if signal_len > max_samples:
            return audio[:, :max_samples]
        
        # If we reach here signal_len < max_samples - so zero pad
        pad_length = max_samples - signal_len
        pad_samples = torch.zeros((1, pad_length))
        audio = torch.cat((audio, pad_samples), 1)

        return audio
    
    '''
    Generates and returns a mel spectrogram for the audio

    n_mels: corresponds to the number of frequency bands (Mel bins). This is the spectrogram height
    fft_size: size of the FFT, creates ``fft_size // 2 + 1`` bins

    Output spectrogram is [channels, n_mels, time] tensor
    '''
    @staticmethod
    def get_mel_spectrogram(audio, sr, n_mels=64, fft_size=1024, hop_length=None):
        transform = transforms.MelSpectrogram(sample_rate=sr, n_fft=fft_size, hop_length=hop_length, n_mels=n_mels)
        spec = transform(audio)
        return spec
