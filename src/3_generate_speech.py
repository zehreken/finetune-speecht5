from rich.traceback import install
install()
"""
Step 3: Generate speech with your cloned voice
"""

import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os

# CONFIGURATION
MODEL_DIR = "speecht5_finetuned/final_model"
EMBEDDING_PATH = "speecht5_finetuned/speaker_embedding.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Step 3: Generating Speech")
print("=" * 60)
print(f"Device: {DEVICE}\n")

# Load model
print("Loading model...")
processor = SpeechT5Processor.from_pretrained(MODEL_DIR)
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_DIR).to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)
print("✓ Model loaded")

# Load speaker embedding
print("Loading speaker embedding...")
speaker_embedding = torch.load(EMBEDDING_PATH, map_location=DEVICE)
print(f"✓ Speaker embedding loaded: {speaker_embedding.shape}\n")

def generate_speech(text, output_path="output.wav"):
    """Generate speech from text"""
    print(f"Generating: '{text}'")
    
    # Process text
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Add batch dimension to embedding
    speaker_emb = speaker_embedding.unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_emb,
            vocoder=vocoder
        )
    
    # Save
    speech_numpy = speech.cpu().numpy()
    sf.write(output_path, speech_numpy, samplerate=16000)
    print(f"✓ Saved to: {output_path}\n")

# Test examples
print("Generating test samples...")
print("-" * 60)

generate_speech(
    "Hello! This is my cloned voice speaking.",
    "test_1.wav"
)

generate_speech(
    "The weather today is absolutely beautiful.",
    "test_2.wav"
)

generate_speech(
    "I enjoy learning about artificial intelligence.",
    "test_3.wav"
)

print("=" * 60)
print("✓ Done! Check the generated .wav files")
print("=" * 60)

# Interactive mode
print("\nInteractive mode - Enter text to generate speech")
print("(Type 'quit' to exit)\n")

counter = 4
while True:
    text = input("Enter text: ").strip()
    
    if text.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not text:
        continue
    
    generate_speech(text, f"generated_{counter}.wav")
    counter += 1