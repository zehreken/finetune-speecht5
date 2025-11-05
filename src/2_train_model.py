from rich.traceback import install
install()
"""
Step 2: Fine-tune SpeechT5 on your voice data
"""

import os
import torch
import numpy as np
import soundfile as sf
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5FeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Dict, List

# CONFIGURATION
DATA_DIR = "my_voice_data"
OUTPUT_DIR = "speecht5_finetuned"
METADATA_FILE = "metadata.txt"
EPOCHS = 100  # More epochs for small dataset
BATCH_SIZE = 2  # Smaller batch size for stability
LEARNING_RATE = 5e-6  # Lower learning rate to prevent collapse
MAX_AUDIO_LENGTH = 10.0  # Maximum audio length in seconds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Step 2: Fine-tuning SpeechT5")
print("=" * 60)
print(f"Device: {DEVICE}\n")

# Load speaker embedding
speaker_embedding = torch.load(os.path.join(OUTPUT_DIR, "speaker_embedding.pt"), map_location='cpu')
print(f"✓ Loaded speaker embedding: {speaker_embedding.shape}\n")

# Load processor and model
print("Loading model...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
feature_extractor = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
print("✓ Model loaded\n")

# Dataset class
class VoiceDataset(Dataset):
    def __init__(self):
        self.samples = []
        
        # Load metadata
        metadata_path = os.path.join(DATA_DIR, METADATA_FILE)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 2:
                    audio_file = parts[0].strip()
                    text = parts[1].strip()
                    audio_path = os.path.join(DATA_DIR, audio_file)
                    
                    if os.path.exists(audio_path):
                        self.samples.append((audio_path, text))
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]
        
        # Load audio using soundfile (no FFmpeg!)
        speech_array, sample_rate = sf.read(audio_path)
        
        # Convert stereo to mono
        if speech_array.ndim > 1:
            speech_array = speech_array.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            num_samples = int(len(speech_array) * 16000 / sample_rate)
            speech_array = signal.resample(speech_array, num_samples)
        
        # Truncate if too long (avoid very long sequences that cause errors)
        max_samples = int(MAX_AUDIO_LENGTH * 16000)
        if len(speech_array) > max_samples:
            speech_array = speech_array[:max_samples]
        
        # Make sure it's float32
        speech_array = speech_array.astype(np.float32)
        
        # Process text
        inputs = processor(text=text, return_tensors="pt", truncation=True, max_length=200)
        
        # Extract mel spectrogram using feature_extractor
        labels = feature_extractor(
            speech_array,
            sampling_rate=16000,
            return_tensors="pt"
        )["input_values"]
        
        # Remove batch dimension
        labels = labels.squeeze(0)
        
        # If it's 2D [n_mels, time], transpose to [time, n_mels]
        if labels.dim() == 2:
            labels = labels.transpose(0, 1)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "speaker_embeddings": speaker_embedding
        }

# Data collator
@dataclass
class DataCollator:
    processor: SpeechT5Processor
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        speaker_embeddings = torch.stack([f["speaker_embeddings"] for f in features])
        
        # Pad input sequences
        max_input_len = max(len(x) for x in input_ids)
        input_ids_padded = torch.stack([
            torch.cat([x, torch.zeros(max_input_len - len(x), dtype=torch.long)])
            for x in input_ids
        ])
        attention_mask_padded = torch.stack([
            torch.cat([x, torch.zeros(max_input_len - len(x), dtype=torch.long)])
            for x in attention_mask
        ])
        
        # Pad labels - handle different possible shapes
        labels_list = []
        for label in labels:
            # If label is 1D, reshape to [time, 80]
            if label.dim() == 1:
                total_elements = label.numel()
                if total_elements % 80 == 0:
                    label = label.reshape(-1, 80)
                else:
                    # Pad to make divisible by 80
                    pad_size = 80 - (total_elements % 80)
                    label = torch.cat([label, torch.zeros(pad_size)])
                    label = label.reshape(-1, 80)
            
            labels_list.append(label)
        
        # Find max time and round UP to nearest multiple of 8
        # This helps avoid alignment issues in SpeechT5
        max_time = max(x.shape[0] for x in labels_list)
        max_time = ((max_time + 7) // 8) * 8  # Round up to nearest 8
        
        labels_padded = []
        for label in labels_list:
            current_time = label.shape[0]
            if current_time < max_time:
                # Pad with zeros on time dimension
                pad_amount = max_time - current_time
                padding = torch.zeros(pad_amount, 80)
                label = torch.cat([label, padding], dim=0)
            labels_padded.append(label)
        
        labels_padded = torch.stack(labels_padded)  # [batch, max_time, 80]
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "speaker_embeddings": speaker_embeddings
        }

# Create dataset
print("\nPreparing dataset...")
dataset = VoiceDataset()

# Split train/eval
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}\n")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=1000,
    save_total_limit=3,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=4,  # Accumulate gradients for stability
    report_to=["none"],
    load_best_model_at_end=False,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,  # Gradient clipping to prevent explosion
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollator(processor=processor),
    tokenizer=processor,
)

# Train!
print("Starting training...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print("-" * 60)

trainer.train()

print("\n✓ Training complete!")

# Save final model
print("\nSaving model...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

print(f"✓ Model saved to {OUTPUT_DIR}/final_model")
print("\nNext step: Run 3_generate_speech.py")