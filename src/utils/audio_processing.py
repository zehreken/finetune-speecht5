"""
Audio Preprocessing for Voice Cloning
Automatically cleans up recordings for optimal training.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
import noisereduce as nr
from tqdm import tqdm

# ============================================================================
# Audio Preprocessing Functions
# ============================================================================

def trim_silence(audio, sample_rate, threshold_db=-40, frame_length=2048, hop_length=512):
    """
    Trim silence from beginning and end, keeping small buffer.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        threshold_db: Volume threshold for silence (lower = more aggressive)
        frame_length: Frame size for analysis
        hop_length: Hop size for analysis
    
    Returns:
        Trimmed audio with small buffers at start/end
    """
    # Find non-silent regions
    intervals = librosa.effects.split(
        audio,
        top_db=abs(threshold_db),
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    if len(intervals) == 0:
        return audio  # Keep original if nothing detected
    
    # Get first and last non-silent points
    start = intervals[0][0]
    end = intervals[-1][1]
    
    # Add small buffer (0.1 seconds) at start and end
    buffer_samples = int(0.1 * sample_rate)
    start = max(0, start - buffer_samples)
    end = min(len(audio), end + buffer_samples)
    
    return audio[start:end]


def normalize_audio(audio, target_db=-20):
    """
    Normalize audio to consistent loudness.
    
    Args:
        audio: Audio waveform
        target_db: Target loudness in dB
    
    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    
    if rms == 0:
        return audio
    
    # Calculate target RMS
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    
    # Apply gain
    normalized = audio * gain
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 0.99:
        normalized = normalized * (0.99 / max_val)
    
    return normalized


def reduce_noise(audio, sample_rate, noise_duration=0.5):
    """
    Reduce background noise from audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        noise_duration: Duration (seconds) of audio to use as noise profile
    
    Returns:
        Noise-reduced audio
    """
    try:
        # Use first portion as noise profile
        noise_sample_length = int(noise_duration * sample_rate)
        noise_sample_length = min(noise_sample_length, len(audio) // 4)
        
        # Reduce noise
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.8
        )
        
        return reduced
    except Exception as e:
        print(f"Warning: Noise reduction failed: {e}")
        return audio


def convert_to_mono(audio):
    """Convert stereo to mono if needed."""
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    return audio


def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio to target sample rate."""
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio


def preprocess_audio_file(
    input_path,
    output_path,
    target_sr=16000,
    trim_silence_enabled=True,
    normalize_enabled=True,
    noise_reduction_enabled=False,  # Can be slow, optional
    silence_threshold=-60
):
    """
    Complete preprocessing pipeline for one audio file.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio
        target_sr: Target sample rate (16000 for SpeechT5)
        trim_silence_enabled: Whether to trim silence
        normalize_enabled: Whether to normalize volume
        noise_reduction_enabled: Whether to reduce noise (slow)
        silence_threshold: Threshold for silence detection
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        
        # Convert to mono
        audio = convert_to_mono(audio)
        
        # Resample if needed
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
        
        # Reduce noise (optional, can be slow)
        if noise_reduction_enabled and len(audio) > sr:  # Only if audio is long enough
            audio = reduce_noise(audio, sr)
        
        # Trim silence
        if trim_silence_enabled:
            audio = trim_silence(audio, sr, threshold_db=silence_threshold)
        
        # Normalize
        if normalize_enabled:
            audio = normalize_audio(audio)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's actually a directory
            os.makedirs(output_dir, exist_ok=True)
        
        # Save
        sf.write(output_path, audio, sr)
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


# ============================================================================
# Batch Processing
# ============================================================================

def preprocess_dataset(
    input_folder="my_voice_data_raw",
    output_folder="my_voice_data",
    target_sr=16000,
    trim_silence_enabled=True,
    normalize_enabled=True,
    noise_reduction_enabled=False,
    silence_threshold=-40
):
    """
    Preprocess all audio files in a folder.
    
    Args:
        input_folder: Folder with raw recordings
        output_folder: Folder for processed audio
        target_sr: Target sample rate
        trim_silence_enabled: Whether to trim silence
        normalize_enabled: Whether to normalize
        noise_reduction_enabled: Whether to reduce noise
        silence_threshold: Silence threshold in dB
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    audio_files = sorted(audio_files)
    
    if len(audio_files) == 0:
        print(f"No audio files found in {input_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"\nProcessing with:")
    print(f"  - Target sample rate: {target_sr} Hz")
    print(f"  - Trim silence: {trim_silence_enabled}")
    print(f"  - Normalize: {normalize_enabled}")
    print(f"  - Noise reduction: {noise_reduction_enabled}")
    print(f"  - Silence threshold: {silence_threshold} dB")
    print()
    
    # Process each file
    success_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Create output filename (always .wav)
        output_filename = audio_file.stem + '.wav'
        output_path = os.path.join(output_folder, output_filename)
        
        # Process
        if preprocess_audio_file(
            input_path=str(audio_file),
            output_path=output_path,
            target_sr=target_sr,
            trim_silence_enabled=trim_silence_enabled,
            normalize_enabled=normalize_enabled,
            noise_reduction_enabled=noise_reduction_enabled,
            silence_threshold=silence_threshold
        ):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count}/{len(audio_files)} files")
    print(f"Output folder: {output_folder}")
    
    # Calculate total duration
    total_duration = 0
    for audio_file in Path(output_folder).glob('*.wav'):
        audio, sr = librosa.load(str(audio_file), sr=None)
        total_duration += len(audio) / sr
    
    print(f"Total audio duration: {total_duration/60:.1f} minutes")
    
    if total_duration < 30 * 60:
        print(f"\n⚠️  Warning: You have less than 30 minutes of audio.")
        print(f"    Recommended: 30-60 minutes for good results")
    elif total_duration > 120 * 60:
        print(f"\n✓ Excellent! You have {total_duration/60:.1f} minutes of audio.")
    else:
        print(f"\n✓ Good amount of training data!")


# ============================================================================
# Quality Check
# ============================================================================

def check_audio_quality(folder="my_voice_data"):
    """
    Check audio quality statistics.
    """
    audio_files = list(Path(folder).glob('*.wav'))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {folder}")
        return
    
    print(f"\n{'='*80}")
    print(f"AUDIO QUALITY REPORT")
    print(f"{'='*80}")
    print(f"Total files: {len(audio_files)}")
    
    durations = []
    sample_rates = []
    max_amplitudes = []
    
    for audio_file in audio_files:
        audio, sr = librosa.load(str(audio_file), sr=None)
        
        durations.append(len(audio) / sr)
        sample_rates.append(sr)
        max_amplitudes.append(np.max(np.abs(audio)))
    
    # Statistics
    durations = np.array(durations)
    
    print(f"\nDuration statistics:")
    print(f"  Mean: {np.mean(durations):.2f} seconds")
    print(f"  Min: {np.min(durations):.2f} seconds")
    print(f"  Max: {np.max(durations):.2f} seconds")
    print(f"  Total: {np.sum(durations)/60:.1f} minutes")
    
    print(f"\nSample rates:")
    unique_srs = set(sample_rates)
    for sr in unique_srs:
        count = sample_rates.count(sr)
        print(f"  {sr} Hz: {count} files")
    
    print(f"\nAmplitude statistics:")
    print(f"  Mean max amplitude: {np.mean(max_amplitudes):.3f}")
    print(f"  Min max amplitude: {np.min(max_amplitudes):.3f}")
    print(f"  Max max amplitude: {np.max(max_amplitudes):.3f}")
    
    # Warnings
    print(f"\n{'='*80}")
    if len(unique_srs) > 1:
        print("⚠️  Warning: Mixed sample rates detected. Consider reprocessing.")
    
    if np.mean(max_amplitudes) < 0.3:
        print("⚠️  Warning: Audio seems quiet. Consider increasing gain.")
    
    if np.max(max_amplitudes) > 0.99:
        print("⚠️  Warning: Some audio may be clipping. Check recordings.")
    
    if np.mean(durations) < 3:
        print("⚠️  Warning: Very short clips. Some might be cut off.")
    
    if np.mean(durations) > 30:
        print("💡 Info: Long clips. Consider splitting for better training.")
    
    print(f"{'='*80}\n")


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess audio for voice cloning")
    parser.add_argument("--input", default="my_voice_data_raw", help="Input folder with raw audio")
    parser.add_argument("--output", default="my_voice_data", help="Output folder for processed audio")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--no-trim", action="store_true", help="Disable silence trimming")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization")
    parser.add_argument("--noise-reduction", action="store_true", help="Enable noise reduction (slow)")
    parser.add_argument("--silence-threshold", type=int, default=-40, help="Silence threshold in dB")
    parser.add_argument("--check-only", action="store_true", help="Only check quality, don't process")
    
    args = parser.parse_args()
    
    if args.check_only:
        check_audio_quality(args.output)
    else:
        # Process audio files
        preprocess_dataset(
            input_folder=args.input,
            output_folder=args.output,
            target_sr=args.sample_rate,
            trim_silence_enabled=not args.no_trim,
            normalize_enabled=not args.no_normalize,
            noise_reduction_enabled=args.noise_reduction,
            silence_threshold=args.silence_threshold
        )
        
        # Check quality
        print("\nChecking processed audio quality...")
        check_audio_quality(args.output)