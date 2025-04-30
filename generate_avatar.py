import whisper
from diffusers import StableDiffusionPipeline
import torch

# 1. Transcribe the audio
model = whisper.load_model("base")
result = model.transcribe("audio_samples/sample.wav", beam_size=5)
prompt = f"cartoon avatar of a person saying: '{result['text']}'"
print("Using prompt:", prompt)

# 2. Generate avatar from prompt
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
)
pipe.to("cpu")  # or "cuda" if GPU is available

image = pipe(prompt).images[0]
image.save("generated_avatar.png")
