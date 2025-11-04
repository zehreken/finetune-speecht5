from rich.traceback import install
install()
"""
Step 1: Extract speaker embeddings from your voice
This creates a single embedding that represents your voice characteristics
"""

import os
import torch
import numpy as np
import soundfile as sf
from scipy import signal

# CONFIGURATION
DATA_DIR = "my_voice_data"
OUTPUT_DIR = "speecht5_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def librosa_mel_basis(sr, n_fft, n_mels):
    """Create mel filterbank (no external dependencies)"""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    # Create mel points
    min_mel = hz_to_mel(0)
    max_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create filterbank
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        for j in range(left, center):
            fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i, j] = (right - j) / (right - center)
    
    return fbank

print("=" * 60)
print("Step 1: Extracting Your Speaker Embedding")
print("=" * 60)
print(f"Device: {DEVICE}\n")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all audio files
audio_files = []
for i in range(1, 201):
    audio_path = os.path.join(DATA_DIR, f"audio_{i:03d}.wav")
    if os.path.exists(audio_path):
        audio_files.append(audio_path)

print(f"Found {len(audio_files)} audio files")

if len(audio_files) == 0:
    print("ERROR: No audio files found!")
    exit(1)

# Extract embeddings from each file
embeddings = []

for i, audio_path in enumerate(audio_files, 1):
    print(f"Processing {i}/{len(audio_files)}: {os.path.basename(audio_path)}", end="\r")
    
    # Load audio using soundfile (no FFmpeg needed!)
    waveform, sample_rate = sf.read(audio_path)
    
    # Convert stereo to mono if needed
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        num_samples = int(len(waveform) * 16000 / sample_rate)
        waveform = signal.resample(waveform, num_samples)
    
    # Convert to tensor
    waveform = torch.from_numpy(waveform).float().to(DEVICE)
    
    # Extract mel spectrogram (manual calculation)
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    
    # Short-time Fourier Transform
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    
    # Power spectrogram
    power_spec = torch.abs(stft) ** 2
    
    # Create mel filterbank
    mel_basis = torch.from_numpy(
        librosa_mel_basis(16000, n_fft, n_mels)
    ).float().to(DEVICE)
    
    # Apply mel filterbank
    mel_spec = torch.matmul(mel_basis, power_spec)
    mel_spec = torch.log(mel_spec + 1e-9).unsqueeze(0)  # Add batch dim
    
    # Compute statistics (mean and std across time)
    mel_mean = torch.mean(mel_spec, dim=2)  # [1, 80]
    mel_std = torch.std(mel_spec, dim=2)    # [1, 80]
    
    # Combine features
    features = torch.cat([mel_mean, mel_std], dim=1)  # [1, 160]
    
    # Expand to 512 dimensions
    embedding = torch.zeros(1, 512).to(DEVICE)
    embedding[:, :160] = features
    
    # Add sinusoidal features for remaining dimensions
    for j in range(160, 512):
        embedding[:, j] = torch.sin(torch.tensor(j / 512.0 * np.pi))
    
    # Normalize
    embedding = embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-8)
    
    embeddings.append(embedding.squeeze(0))

print("\n")

# Average all embeddings
avg_embedding = torch.stack(embeddings).mean(dim=0)
avg_embedding = avg_embedding / (torch.norm(avg_embedding) + 1e-8)

# Save
output_path = os.path.join(OUTPUT_DIR, "speaker_embedding.pt")
torch.save(avg_embedding, output_path)

print(f"✓ Speaker embedding extracted: shape {avg_embedding.shape}")
print(f"✓ Saved to: {output_path}")
print("\nNext step: Run 2_train_model.py")