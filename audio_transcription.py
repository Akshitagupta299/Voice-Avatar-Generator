from faster_whisper import WhisperModel

# Load the model (this downloads a much smaller model!)
model = WhisperModel("small")

# Transcribe the audio
segments, info = model.transcribe("C:\\Users\\JMD\\OneDrive\\Documents\\Shared Folder\\Projects\\Voice-Avatar-Generator\\audio_samples\\sample.wav")

# Print result
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
