# Voice-Avatar Generator 🎙️🧑‍💻

Transform your **voice or text prompts into custom avatars** in different artistic styles such as Anime, Pixar, or Ghibli — all from a sleek and interactive web interface!

---

## 🔥 Features

- 🎤 **Voice Input**: Upload an audio file and automatically convert it into a descriptive text using OpenAI Whisper.
- ✍️ **Text Prompt**: Manually enter any custom prompt to describe the avatar.
- 🎨 **Choose Style**: Select from multiple artistic styles like Anime, Pixar, and more.
- 🧠 **AI Avatar Generation**: Uses Stable Diffusion to generate avatars from text.
- 💾 **Download Button**: Save your generated avatar easily.
- 🌐 **User-Friendly UI**: Built using Flask and Bootstrap for smooth interaction..

---

## 🛠️ Setup and Run

### 1. Clone the Repository
git clone https://github.com/your-username/voice-avatar-generator.git
cd voice-avatar-generator

### 2. Create and Activate Virtual Environment (Optional but Recommended)
python -m venv venv
# For macOS/Linux:
source venv/bin/activate
# For Windows:
venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the App
python main.py
You can customize main.py to:
a. Accept an audio file
b. Generate avatar animation
c. Display or save the output video

---

## 🛠️ Tech Stack
### Frontend: HTML, CSS, JavaScript, Bootstrap
### Backend: Flask (Python)
### Models:
a. OpenAI Whisper – for speech-to-text conversion
b. Stable Diffusion – for avatar generation

---

## 📌 To-Do / Future Enhancements
a. 🗣️ Add lip-sync animation (mouth movement)
b. 🌍 Add multilingual support for voice input
c. 🧵 Allow avatar customization (hairstyle, accessories)
d. 🎬 Support for short animated avatar clips

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## 📄 License
This project is open-source and available under the MIT License.

## 🙌 Acknowledgements
- OpenAI for Whisper
- Hugging Face for Stable Diffusion pipeline
- FFmpeg for media processing
- Flask for backend support

---

🙋‍♀️ Created by
Akshita Gupta
Final Year B.Tech, AI & DS
Lakshmi Narain College of Technology, Bhopal
