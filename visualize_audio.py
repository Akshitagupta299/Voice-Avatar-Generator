import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    plt.figure(figsize=(14, 8))

    # 1. Plot waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")

    # 2. Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram (dB)")

    # 3. Plot MFCCs
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title("MFCCs")

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    visualize_audio("sample.wav")
