from rich.traceback import install
install()

"""
SpeechT5 Voice Cloning Inference Script
Generate speech using your fine-tuned voice model
"""

import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pathlib import Path

class VoiceCloner:
    """Generate speech with your cloned voice"""
    
    def __init__(self, model_dir: str = "speecht5_finetuned", device: str = "cuda"):
        """
        Initialize the voice cloner
        
        Args:
            model_dir: Directory containing fine-tuned model and speaker embedding
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load processor
        print("Loading processor...")
        self.processor = SpeechT5Processor.from_pretrained(
            f"{model_dir}/final_model"
        )
        
        # Load fine-tuned model
        print("Loading fine-tuned model...")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            f"{model_dir}/final_model"
        ).to(self.device)
        
        # Load vocoder
        print("Loading vocoder...")
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)
        
        # Load your speaker embedding
        print("Loading your speaker embedding...")
        embedding_path = Path(model_dir) / "speaker_embedding.pt"
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Speaker embedding not found at {embedding_path}. "
                "Make sure you've completed training first!"
            )
        
        self.speaker_embedding = torch.load(embedding_path).to(self.device)
        
        print("✓ Voice cloner initialized successfully!")
    
    def generate_speech(
        self,
        text: str,
        output_path: str = "output.wav",
        sample_rate: int = 16000
    ):
        """
        Generate speech from text using your cloned voice
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the generated audio file
            sample_rate: Output sample rate (default: 16000 Hz)
        
        Returns:
            Path to the generated audio file
        """
        print(f"\nGenerating speech for: '{text}'")
        
        try:
            # Process text
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Add batch dimension to speaker embedding if needed
            speaker_emb = self.speaker_embedding.unsqueeze(0) if self.speaker_embedding.dim() == 1 else self.speaker_embedding
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    speaker_emb,
                    vocoder=self.vocoder
                )
            
            # Convert to numpy and save
            speech_numpy = speech.cpu().numpy()
            sf.write(output_path, speech_numpy, samplerate=sample_rate)
            
            print(f"✓ Speech saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ Error generating speech: {e}")
            raise
    
    def generate_batch(
        self,
        texts: list,
        output_dir: str = "generated_speech",
        prefix: str = "speech"
    ):
        """
        Generate speech for multiple texts
        
        Args:
            texts: List of texts to convert to speech
            output_dir: Directory to save generated audio files
            prefix: Prefix for output filenames
        
        Returns:
            List of paths to generated audio files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        for i, text in enumerate(texts, 1):
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.wav")
            self.generate_speech(text, output_path)
            output_paths.append(output_path)
        
        print(f"\n✓ Generated {len(output_paths)} audio files in {output_dir}/")
        return output_paths

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the voice cloner"""
    
    print("=" * 80)
    print("SpeechT5 Voice Cloning - Inference")
    print("=" * 80)
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        model_dir="speecht5_finetuned",
        device="cuda"  # Use "cpu" if no GPU available
    )
    
    # Example 1: Generate single speech
    print("\n" + "-" * 80)
    print("Example 1: Single Text Generation")
    print("-" * 80)
    
    text = "Hello! This is my cloned voice speaking. How do I sound?"
    cloner.generate_speech(text, output_path="my_voice_test.wav")
    
    # Example 2: Generate multiple speeches
    print("\n" + "-" * 80)
    print("Example 2: Batch Generation")
    print("-" * 80)
    
    texts = [
        "The weather today is absolutely beautiful.",
        "I enjoy learning about artificial intelligence and machine learning.",
        "Voice cloning technology has made remarkable progress in recent years.",
        "This is a test of the speech synthesis system.",
        "Thank you for listening to my cloned voice!"
    ]
    
    cloner.generate_batch(
        texts,
        output_dir="generated_samples",
        prefix="sample"
    )
    
    # Example 3: Interactive mode
    print("\n" + "-" * 80)
    print("Example 3: Interactive Mode")
    print("-" * 80)
    print("Enter text to generate speech (or 'quit' to exit)")
    
    counter = 1
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
        
        try:
            output_path = f"interactive_{counter:03d}.wav"
            cloner.generate_speech(user_input, output_path)
            counter += 1
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()