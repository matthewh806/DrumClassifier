from torch.utils.data import DataLoader, Dataset
import torchaudio
from classifier.audioutils import AudioUtils

class SoundDataset(Dataset):
    '''
    Initialises the SoundDataset

    audio_data should be a list of tuples (audio_path, label)
    len samples is the desired length in samples to fix the audio files to by padding or truncation
    '''
    def __init__(self, audio_datas, channels, len_samples, sr=44100):
        self.audio_datas = audio_datas
        self.channels = channels
        self.sample_rate = sr
        self.len_samples = len_samples

    def __len__(self):
        return len(self.audio_datas)

    '''
    Gets the mel audio spectrogram and class_id (label) for the audio file at idx

    This performs transformations on the audio data to ensure its in the format we expect
    in terms of number of channels & size
    '''
    def __get__item(self, idx):
        audio_data = self.audio_datas[idx]
        audio_path = audio_data[0]
        class_id = audio_path[1]

        signal, sr = AudioUtils.open(audio_path)
        signal = AudioUtils.convert_mono(signal)
        signal = AudioUtils.resize(self.len_samples)

        mel_spectrogram = AudioUtils.get_mel_spectrogram(signal, self.sample_rate, hop_length=128, n_mels=64)

        return mel_spectrogram, class_id