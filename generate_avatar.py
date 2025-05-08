import whisper
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Load once (globally)
WHISPER_MODEL = whisper.load_model("base")

def transcribe_audio(audio_path):
    """Transcribe the uploaded audio file using Whisper."""
    try:
        result = WHISPER_MODEL.transcribe(audio_path, beam_size=5)
        return result["text"]
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None

def get_prompt_from_style(user_style):
    """Return a Stable Diffusion prompt based on selected style."""
    styles = {
        "cartoon": "Cartoon avatar of a calm young woman, soft facial features, light smile, clear background, close-up",
        "anime": "Anime-style portrait of a calm young woman, big eyes, clean lines, soft colors, head and shoulders, light smile",
        "Pixar": "Pixar-style 3D character portrait of a calm young woman, detailed face, warm smile, soft lighting, cinematic style",
        "sketch": "Hand-drawn pencil sketch of a calm young woman, clean lines, minimal shading, soft smile, centered face",
        "realistic": "Hyper-realistic digital portrait of a calm young woman, soft expression, photographic quality",
        "ghibli": "Studio Ghibli-style character, calm young woman with expressive eyes, gentle smile, hand-drawn animation style, warm tones"
    }
    return styles.get(user_style, styles["cartoon"])

def load_diffusion_pipeline():
    """Load the Stable Diffusion pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32
    )
    pipe.to("cpu")  # Use 'cuda' if GPU is available
    return pipe

def generate_avatar_image(prompt, pipeline, user_style):
    """Generate the avatar image and return the file path."""
    try:
        image = pipeline(prompt, num_inference_steps=35, guidance_scale=8.5).images[0]
        os.makedirs("avatars", exist_ok=True)
        image_path = f"avatars/generated_avatar_{user_style}.png"
        image.save(image_path)
        return image_path
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None

# -----------------------
# Main flow (for testing)
# -----------------------
if __name__ == "__main__":
    audio_path = "audio_samples/sample.wav"
    user_style = "ghibli"

    print("🔊 Transcribing audio...")
    text_prompt = transcribe_audio(audio_path)
    if not text_prompt:
        print("❌ Failed to transcribe audio.")
        exit()

    print("🎨 Getting style prompt...")
    style_prompt = get_prompt_from_style(user_style)

    full_prompt = f"{style_prompt}, {text_prompt}"
    print(f"🧠 Final Prompt: {full_prompt}")

    print("🧪 Loading image generation pipeline...")
    pipeline = load_diffusion_pipeline()

    print("🖼️ Generating avatar...")
    avatar_path = generate_avatar_image(full_prompt, pipeline, user_style)

    if avatar_path:
        print(f"✅ Avatar saved at {avatar_path}")
    else:
        print("❌ Avatar generation failed.")
