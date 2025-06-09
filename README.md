# Picture Prompt Storytelling (Hugging Face Demo)
A Hugging Face Space demo that takes an uploaded image, uses a vision‐to‐text model to describe it, feeds the description into ChatGPT to generate a fun story, then converts the story to speech with Microsoft’s SpeechT5 TTS model.


[▶️ Try it live on Hugging Face Spaces](https://huggingface.co/spaces/yh24hsu/picture_to_voicestory_demo)

---

## 🚀 Features

- **Image Captioning**  
  Uses Salesforce’s `blip-image-captioning-base` to turn any uploaded picture into a short text description.

- **Story Generation**  
  Prompts OpenAI’s GPT-4o model (via LangChain) to spin the description into a 100-word B1-level language-learner-friendly story.

- **Text‐to‐Speech**  
  Leverages Microsoft’s SpeechT5 (with a pretrained HiFi-GAN vocoder) to produce a natural voice-over of the story.

- **One‐Click Web UI**  
  Built on Gradio for an intuitive “Upload → Submit → Play” experience entirely in your browser.

---

## 📦 Requirements

List of Python packages (pinned versions in `requirements.txt`):

```text
gradio
transformers
torch
soundfile
datasets
sentencepiece
langchain-core
langchain-openai
openai
pillow

