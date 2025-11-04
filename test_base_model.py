"""
Test the base SpeechT5 model with your speaker embedding
This will tell us if the issue is with training or the embedding
"""

import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Testing BASE SpeechT5 model with your speaker embedding")
print(f"Device: {DEVICE}\n")

# Load BASE model (not fine-tuned)
print("Loading base model...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)
print("✓ Base model loaded")

# Load your speaker embedding
print("Loading your speaker embedding...")
speaker_embedding = torch.load("speecht5_finetuned/speaker_embedding.pt", map_location=DEVICE)
print(f"✓ Embedding loaded: {speaker_embedding.shape}\n")

# Test with base model
text = "Hello, this is a test of the base model with your voice embedding."
print(f"Generating: '{text}'")

inputs = processor(text=text, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

speaker_emb = speaker_embedding.unsqueeze(0)

with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_emb,
        vocoder=vocoder
    )

speech_numpy = speech.cpu().numpy()
sf.write("test_base_model.wav", speech_numpy, samplerate=16000)

print("✓ Saved to: test_base_model.wav")
print("\nListen to this file. If it sounds like normal speech,")
print("then the issue is with training. If it's also noise,")
print("then the issue is with the speaker embedding.")