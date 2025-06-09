import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from transformers import pipeline
#LLMChain 連接多個預測模型  OpenAI 掉用街口
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import gradio as gr 
from PIL import Image

import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
# if openai_key is None:
#     raise ValueError("Please set the OPENAI_API_KEY environment variable.")


#image to text 
def img2text(url):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img_to_text(url)[0]["generated_text"]
    
    print("text: " + text)
    return text

#lls
def generate_story(scenario):
    
    template = """You are like a grandmom who is good at telling stories. the CONTEXT: {scenario} below is the 
    paragraph of a picture, please use this paragraph to generate a fun, funny and interesting story. 
    The story is around 80 words long, and it is suitable for the laguage learner with LEVEL
    
    CONTEXT: {scenario}
    LEVEL: B1
    """
    
    prompt = PromptTemplate(template = template , input_variables=["scenario"])
    
    story_llm = LLMChain(llm=ChatOpenAI(model_name="gpt-4o",temperature=0.9), prompt=prompt)
    
    
    story = story_llm.predict(scenario=scenario)
    print("story: " + story)
    return story


# init SpeechT5 model and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def text2speech(text, filename="story.wav"):
    max_len = 600
    inputs = processor(
        text=text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len
    )
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    sf.write(filename, speech.numpy(), samplerate=16000)
    print(f"✅ Audio saved as {filename}")
    return filename


# text = img2text("pic.png")
# story = generate_story(text)
# tmp_path = text2speech(story)


#  Gradio interface
def image_to_story_tts(image: Image.Image):
    try:
        scen = img2text(image)
        story = generate_story(scen)
        audio = text2speech(story)
        return scen, story, audio
    except Exception as e:
        err = f"Error: {e}"
        return err, err, None


#Gradio creates frontend
title = "Picture → Text Story → Voice Demo"
description = "uploda a picture ; chat GPT generates a text story ; TTS transfers to voice"

iface = gr.Interface(
    fn=image_to_story_tts,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Image-to-Text"), 
        gr.Textbox(label="Story Text"), 
        gr.Audio(label="Generated Voice Story")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()