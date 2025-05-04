import whisper
from diffusers import StableDiffusionPipeline
import torch

# 1. Transcribe the audio
model = whisper.load_model("base")
result = model.transcribe("audio_samples/sample.wav", beam_size=5)

# Style definitions
styles = {
    "cartoon": "Cartoon avatar of a calm young woman, soft facial features, light smile, clear background, close-up",
    "anime": "Anime-style portrait of a calm young woman, big eyes, clean lines, soft colors, head and shoulders, light smile",
    "Pixar": "Pixar-style 3D character portrait of a calm young woman, detailed face, warm smile, soft lighting, cinematic style",
    "sketch": "Hand-drawn pencil sketch of a calm young woman, clean lines, minimal shading, soft smile, centered face",
    "realistic": "Hyper-realistic digital portrait of a calm young woman, soft expression, photographic quality",
    "ghibli": "Studio Ghibli-style character, calm young woman with expressive eyes, gentle smile, hand-drawn animation style, warm tones"
}

# 2. Select style (this would come from the user in a real app)
user_choice = "ghibli"  # Change this to test different styles or get from UI input
prompt = styles.get(user_choice, styles["cartoon"])  # default to cartoon if invalid
print(f"Using style: {user_choice}")
print(f"Using prompt: {prompt}")

# 3. Generate avatar from prompt
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
)
pipe.to("cpu")  # Or use "cuda" if you have a GPU

image = pipe(prompt, num_inference_steps=35, guidance_scale=8.5).images[0]

# 4. Save image
image_path = f"avatars/generated_avatar_{user_choice}.png"
image.save(image_path)
print(f"Avatar saved to {image_path}")
