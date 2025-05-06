from flask import Flask, request, jsonify, render_template, send_file
import os
import whisper
from diffusers import StableDiffusionPipeline
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base")

# Define style prompts
STYLE_PROMPTS = {
    "anime": "anime-style portrait of ",
    "pixar": "Pixar-style 3D character of ",
    "ghibli": "Studio Ghibli character of ",
    "realistic": "photo-realistic portrait of ",
    "cartoon": "cartoon drawing of ",
    "sketch": "hand-drawn pencil sketch of "
}

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/generate-avatar', methods=['POST'])
def generate_avatar():
    prompt_text = request.form.get('prompt')
    style = request.form.get('style')
    audio = request.files.get('audio')

    # Use audio transcription if no text prompt is provided
    if not prompt_text and audio:
        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio.filename))
        audio.save(audio_path)
        result = whisper_model.transcribe(audio_path)
        prompt_text = result["text"]

    if not prompt_text:
        return jsonify({"error": "Please provide a prompt or audio"}), 400

    full_prompt = STYLE_PROMPTS.get(style, "") + prompt_text

    image = pipe(prompt=full_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image_path = os.path.join(GENERATED_FOLDER, "generated_avatar.png")
    image.save(image_path)

    return send_file(image_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
