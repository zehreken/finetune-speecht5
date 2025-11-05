"""
This is to test speecht5 text to speech model
It runs on the GPU, low quality but quite fast
"""

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np

# Load models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to("cuda")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to("cuda")

# Generate speech
text = "If someone is able to show me that what I think or do is not right, I will happily change, for I seek the truth, by which no one was ever truly harmed. It is the person who continues in his self-deception and ignorance who is harmed. silencio"
inputs = processor(text=text, return_tensors="pt").to("cuda")

# Use a simple speaker embedding (512-dim vector)
# This is just a random one - you can save/reuse specific voices
torch.manual_seed(42)
speaker_embeddings = torch.randn(1, 512).to("cuda")

# Generate
speech = model.generate_speech(
    inputs["input_ids"], 
    speaker_embeddings, 
    vocoder=vocoder,
    minlenratio=0.0,  # Allow shorter outputs
    maxlenratio=20.0  # Allow longer outputs (increase if still cutting off)
)

# Save
sf.write("output/voice_test.wav", speech.cpu().numpy(), samplerate=16000)
print("Audio saved to output.wav")