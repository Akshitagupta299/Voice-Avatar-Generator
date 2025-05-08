import os
from flask import Flask, render_template, request, send_from_directory
import whisper
from diffusers import StableDiffusionPipeline
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Style prompts
STYLES = {
    "cartoon": "Cartoon avatar of a calm young woman, soft facial features, light smile, clear background, close-up",
    "anime": "Anime-style portrait of a calm young woman, big eyes, clean lines, soft colors, head and shoulders, light smile",
    "Pixar": "Pixar-style 3D character portrait of a calm young woman, detailed face, warm smile, soft lighting, cinematic style",
    "sketch": "Hand-drawn pencil sketch of a calm young woman, clean lines, minimal shading, soft smile, centered face",
    "realistic": "Hyper-realistic digital portrait of a calm young woman, soft expression, photographic quality",
    "ghibli": "Studio Ghibli-style character, calm young woman with expressive eyes, gentle smile, hand-drawn animation style, warm tones"
}

# Load Whisper once
whisper_model = whisper.load_model("base")

# Load Stable Diffusion once
diffusion_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
)
diffusion_pipe.to("cpu")  # or 'cuda' if GPU is available


def transcribe_audio(audio_path):
    """Transcribes the audio and returns the text"""
    result = whisper_model.transcribe(audio_path, beam_size=5)
    return result['text']


def get_prompt(style_choice, text_prompt):
    """Combines style with user prompt or transcribed text"""
    style = STYLES.get(style_choice, STYLES["cartoon"])
    return f"{style}. {text_prompt}"


def generate_avatar(prompt, style_choice):
    """Generates image using Stable Diffusion and saves it"""
    image = diffusion_pipe(prompt, num_inference_steps=35, guidance_scale=8.5).images[0]
    image_filename = f"avatar_{style_choice}.png"
    image_path = os.path.join(GENERATED_FOLDER, image_filename)
    image.save(image_path)
    return image_filename


@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None
    message = None
    error = None

    if request.method == "POST":
        try:
            text_prompt = request.form.get("prompt", "").strip()
            style_choice = request.form.get("style", "cartoon")
            audio_file = request.files.get("audio")

            if audio_file and audio_file.filename:
                filename = secure_filename(audio_file.filename)
                audio_path = os.path.join(UPLOAD_FOLDER, filename)
                audio_file.save(audio_path)
                text_prompt = transcribe_audio(audio_path)
                message = "Avatar generation started. This may take a few seconds..."

            if not text_prompt:
                error = "Please provide a text prompt or upload an audio file."
            else:
                prompt = get_prompt(style_choice, text_prompt)
                image_filename = generate_avatar(prompt, style_choice)
                image_url = f"/{GENERATED_FOLDER}/{image_filename}"

        except Exception as e:
            print("Error:", e)
            error = "Something went wrong while generating the avatar. Please try again."

    return render_template("index.html", image_url=image_url, message=message, error=error)


@app.route('/generated/<filename>')
def serve_generated_image(filename):
    return send_from_directory(GENERATED_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
