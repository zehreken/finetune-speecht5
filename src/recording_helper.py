"""
Interactive Recording Helper
Run this while recording to see each sentence one at a time.
"""

import os
import time

def display_recording_prompts(script_file="recording_script.txt"):
    """Display sentences one by one with timing."""
    
    sentences = []
    
    # Read sentences from script
    with open(script_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('|')
                if len(parts) == 2:
                    filename, text = parts
                    sentences.append((filename, text))
    
    print("=" * 80)
    print("VOICE RECORDING SESSION")
    print("=" * 80)
    print(f"Total sentences to record: {len(sentences)}")
    print("\nInstructions:")
    print("- Press ENTER when ready to see the next sentence")
    print("- Start recording when you see the sentence")
    print("- Stop recording after speaking")
    print("- Save as the filename shown")
    print("- Take breaks every 20-30 recordings")
    print("=" * 80)
    
    input("\nPress ENTER to start...")
    
    for i, (filename, text) in enumerate(sentences, 1):
        print(f"\n{'='*80}")
        print(f"Recording {i}/{len(sentences)}")
        print(f"{'='*80}")
        print(f"\nFilename: {filename}")
        print(f"\n>>> {text}")
        print(f"\n{'='*80}")
        
        # Suggest breaks
        if i % 25 == 0 and i < len(sentences):
            print(f"\n*** SUGGESTION: Take a short break! You've done {i} recordings. ***")
            input("Press ENTER when ready to continue...")
        else:
            input("Press ENTER for next sentence...")
    
    print(f"\n{'='*80}")
    print("🎉 ALL DONE! You've completed all recordings!")
    print(f"{'='*80}")
    print(f"\nTotal recordings: {len(sentences)}")
    print("\nNext steps:")
    print("1. Check that all audio files are properly saved")
    print("2. Listen to a few random recordings to verify quality")
    print("3. Run the training script with your data!")

if __name__ == "__main__":
    display_recording_prompts()
