import torch
import soundfile as sf
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)

# ========== CONFIGURATION ==========
TEXT = """Hello, I'm Guchan. I'm a game programmer with 16 years of experience. I've worked on PC, console and mobile games.
I'm a very curious person by nature. I like thinking in first principles, breaking things down to simpler parts and build again.
I think I'm a great fit for this position on either intermediate or senior level. Given the opportunity I would gladly step up to help building the team and leading afterwards.
Here you are, listening to my cloned voice using Speecht5 model. I thought it would be a fun idea to send my cover letter for this position
in audio format using my cloned voice but I didn't realize how time consuming it would be even to achieve this horrible cloned voice."""
texts = TEXT.split(".")
SPEAKER_EMBED_PATH = "speecht5_finetuned/speaker_embedding.pt"
OUTPUT_PATH = "output.wav"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== LOAD MODELS ==========
print("Loading models...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

# ========== LOAD SPEAKER EMBEDDING ==========
print("Loading speaker embedding:", SPEAKER_EMBED_PATH)
speaker_embeddings = torch.load(SPEAKER_EMBED_PATH, map_location=DEVICE)
if speaker_embeddings.ndim == 1:
    speaker_embeddings = speaker_embeddings.unsqueeze(0)
assert speaker_embeddings.shape[-1] == 512, "Speaker embedding must be 512-dim"

# ========== TEXT TO SPEECH ==========
index = 0
for text in texts:
    print("Generating speech...")
    inputs = processor(text=text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=vocoder,
        )

    # ========== SAVE OUTPUT ==========
    output_path = "output_" + str(index) + ".wav"
    sf.write(output_path, speech.cpu().numpy(), samplerate=SAMPLE_RATE)
    index += 1
    print(f"Done! Saved synthesized speech to {output_path}")
print("Success!")
