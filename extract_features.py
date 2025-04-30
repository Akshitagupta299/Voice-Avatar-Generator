import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def extract_voice_features(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=16000)  # Load at 16kHz
    print(f"Audio loaded: {len(y)/sr:.2f} seconds")

    # 1. Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # 2. Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # 3. Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)

    # 4. Extract Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    # 5. Extract Energy
    energy = np.mean(librosa.feature.rms(y=y))

    return {
        "mfccs": mfccs_mean,
        "chroma": chroma_mean,
        "mel": mel_mean,
        "pitch": pitch,
        "energy": energy
    }

# Example usage
if __name__ == "__main__":
    file_path = "C:\\Users\\JMD\\OneDrive\\Documents\\Shared Folder\\Projects\\Voice-Avatar-Generator\\audio_samples\\sample.wav"  # Replace with your own .wav file
    features = extract_voice_features(file_path)

    print("\nExtracted Features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: Mean shape = {value.shape}")
        else:
            print(f"{key}: {value:.3f}")
