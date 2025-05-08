# Voice-Avatar Generator 🎙️🧑‍💻
![Voice-Avatar Generator](https://img.shields.io/badge/Language-Python-blue.svg) ![Status](https://img.shields.io/badge/Status-Active-green.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Transform your **voice or text prompts into custom avatars** in different artistic styles such as Anime, Pixar, or Ghibli — all from a sleek and interactive web interface!

## 🔥 Features

- 🎤 **Voice Input**: Upload an audio file and automatically convert it into a descriptive text using OpenAI Whisper.
- ✍️ **Text Prompt**: Manually enter any custom prompt to describe the avatar.
- 🎨 **Choose Style**: Select from multiple artistic styles like Anime, Pixar, and more.
- 🧠 **AI Avatar Generation**: Uses Stable Diffusion to generate avatars from text.
- 💾 **Download Button**: Save your generated avatar easily.
- 🌐 **User-Friendly UI**: Built using Flask and Bootstrap for smooth interaction..

## 🛠️ Setup and Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/voice-avatar-generator.git
cd voice-avatar-generator
```

### 2️⃣ Create and Activate Virtual Environment (Optional but Recommended)
``` bash
python -m venv venv
```
# For macOS/Linux:
```
source venv/bin/activate
```
# For Windows:
```
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
``` bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
``` bash
python app.py
```
The application will be available at `http://127.0.0.1:5000/`.

## 📂Project WorkFlow
```
User Interface (Web Browser)
│
▼
Flask Backend Server (Python)
├── Audio/Text Input Handler
├── Whisper Speech-to-Text Module
├── Prompt Preprocessor (with Style Modifiers)
├── Stable Diffusion Image Generator
├── Response Handler
└── Image Renderer & Download Support
│
▼
AI Models & Files (Whisper, Diffusers, etc.)

```

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Backend:** Flask (Python)
- **Models:**
- OpenAI Whisper – for speech-to-text conversion
- Stable Diffusion – for avatar generation


## 📌 To-Do / Future Enhancements
- 🗣️ Add lip-sync animation (mouth movement)
- 🌍 Add multilingual support for voice input
- 🧵 Allow avatar customization (hairstyle, accessories)
- 🎬 Support for short animated avatar clips

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## 📄 License
This project is licensed under the [MIT License](./LICENSE).  
© 2025 Akshita Gupta

## 🙌 Acknowledgements
- OpenAI for Whisper
- Hugging Face for Stable Diffusion pipeline
- FFmpeg for media processing
- Flask for backend support

## 🙋‍♀️ Created by
 **Akshita Gupta**
