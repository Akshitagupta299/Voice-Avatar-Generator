from pyAudioAnalysis import audioTrainTest as aT

def predict_emotion(audio_path):
    result, _, _ = aT.file_classification(audio_path, "emotion_model/svmModel", "svm")
    emotions = ['neutral', 'happy', 'sad', 'angry']  # Customize as per your training
    predicted = emotions[int(result)]
    print(f"Predicted Emotion: {predicted}")
    return predicted

# Add this block to test the function
if __name__ == "__main__":
    audio_path = "C:\\Users\\JMD\\OneDrive\\Documents\\Shared Folder\\Projects\\Voice-Avatar-Generator\\audio_samples\\sample.wav"  # <-- Replace with your actual .wav file path
    predict_emotion(audio_path)
