"""
Voice Recording Dataset Generator
Creates metadata file with diverse sentences optimized for voice cloning.
Then helps you record them systematically.
"""

import os
import random
from datetime import datetime

# ============================================================================
# PART 1: Generate Diverse Training Sentences
# ============================================================================

def generate_metadata_file(output_file="recording_script.txt", num_sentences=200):
    """
    Generate a diverse set of sentences for voice recording.
    Includes phonetically rich content, varied emotions, and different speaking styles.
    """
    
    # Categories of sentences for comprehensive voice coverage
    
    # 1. Phonetically rich sentences (cover all English sounds)
    phonetic_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "Sphinx of black quartz, judge my vow.",
        "The five boxing wizards jump quickly.",
        "Jackdaws love my big sphinx of quartz.",
        "Mr. Jock, TV quiz PhD, bags few lynx.",
        "Waltz, bad nymph, for quick jigs vex.",
    ]
    
    # 2. Common everyday phrases
    everyday_phrases = [
        "Hello, how are you doing today?",
        "That sounds like a great idea!",
        "I'm not sure I understand what you mean.",
        "Could you please repeat that?",
        "Thank you so much for your help.",
        "I really appreciate your time.",
        "Let me think about that for a moment.",
        "That's interesting, tell me more.",
        "I completely agree with you.",
        "I see what you're saying.",
        "Would you like some coffee or tea?",
        "The weather is beautiful today.",
        "What time does the meeting start?",
        "I'll get back to you as soon as possible.",
        "Have a wonderful day!",
    ]
    
    # 3. Technical and formal speech
    technical_sentences = [
        "The algorithm optimizes performance through parallel processing.",
        "We need to analyze the data before making any conclusions.",
        "The implementation follows industry best practices.",
        "Please refer to the documentation for detailed instructions.",
        "The system architecture consists of multiple microservices.",
        "We should schedule a meeting to discuss the requirements.",
        "The project deadline is next Friday afternoon.",
        "According to recent research, the results are promising.",
        "The configuration file needs to be updated regularly.",
        "Let's review the specifications one more time.",
    ]
    
    # 4. Emotional and expressive sentences
    emotional_sentences = [
        "That's absolutely amazing! I can't believe it!",
        "I'm so sorry to hear about that.",
        "This is incredibly frustrating.",
        "I'm really excited about this opportunity!",
        "That makes me feel a bit uncomfortable.",
        "I'm genuinely happy for you!",
        "This situation is quite concerning.",
        "What a pleasant surprise!",
        "I'm disappointed with the outcome.",
        "That's hilarious! I can't stop laughing!",
    ]
    
    # 5. Questions (important for intonation)
    questions = [
        "What do you think about this?",
        "Can you help me with something?",
        "Where did you put my keys?",
        "Why is this happening?",
        "How does this work exactly?",
        "When will you be available?",
        "Who was responsible for this decision?",
        "Which option do you prefer?",
        "Are you sure about that?",
        "Could this be improved somehow?",
    ]
    
    # 6. Long sentences (for sustained speech)
    long_sentences = [
        "I've been thinking about this problem for a while now, and I believe the best solution would be to break it down into smaller, more manageable tasks that we can tackle one at a time.",
        "When I was younger, I used to spend hours reading books in the library, completely losing track of time and immersing myself in different worlds and stories.",
        "The conference will be held next month in Stockholm, and we're expecting participants from all over Europe to attend and share their latest research findings.",
        "Despite the challenges we've faced this year, I'm optimistic that we can overcome them through teamwork, dedication, and innovative thinking.",
        "If you're interested in learning more about this topic, I would recommend starting with the introductory materials and then gradually working your way up to the more advanced concepts.",
    ]
    
    # 7. Short responses (natural conversation)
    short_responses = [
        "Yes, exactly.",
        "Not really.",
        "Sure thing!",
        "I don't think so.",
        "That's right.",
        "Maybe later.",
        "Of course.",
        "Not at all.",
        "Absolutely.",
        "I suppose so.",
        "Fair enough.",
        "Good point.",
        "That works.",
        "No problem.",
        "Sounds good.",
    ]
    
    # 8. Numbers and dates (important for practical use)
    numbers_dates = [
        "The meeting is scheduled for March fifteenth at three thirty PM.",
        "Please call me at five five five, one two three four.",
        "The total comes to one hundred and forty-seven dollars and fifty cents.",
        "I was born on the twenty-third of August, nineteen ninety-five.",
        "The address is twelve thirty-four Main Street, apartment five B.",
        "We need approximately two thousand five hundred units.",
        "The deadline is December thirty-first at midnight.",
        "My passport number is A one two three four five six seven eight nine.",
    ]
    
    # 9. Varied punctuation and pauses
    punctuation_sentences = [
        "Well, let me see... I think that could work.",
        "Wait! Don't forget your jacket!",
        "The options are: first, we could wait; second, we could proceed; or third, we could cancel.",
        "She said - and I quote - 'This is unacceptable.'",
        "Really? You're saying that's what happened?",
        "The ingredients include: flour, sugar, eggs, and butter.",
        "However, there's one thing I should mention first.",
        "In other words, we need to act quickly.",
    ]
    
    # 10. Accent-highlighting words (since user is in Sweden)
    accent_words = [
        "I need to schedule an appointment for next Thursday.",
        "The water bottle is on the table next to the computer.",
        "We're going to visit the castle this weekend.",
        "I'd rather have coffee than tea, if that's alright.",
        "The weather forecast predicts rain throughout the week.",
        "I thought about it thoroughly before making a decision.",
        "The organization is headquartered in the city center.",
        "I'm particularly interested in this opportunity.",
    ]
    
    # Combine all categories
    all_sentences = (
        phonetic_sentences * 3 +  # More of these for phonetic coverage
        everyday_phrases * 2 +
        technical_sentences * 2 +
        emotional_sentences * 2 +
        questions * 2 +
        long_sentences * 2 +
        short_responses * 3 +  # Many short ones for natural conversation
        numbers_dates * 2 +
        punctuation_sentences * 2 +
        accent_words * 2
    )
    
    # Shuffle and select desired number
    random.shuffle(all_sentences)
    selected_sentences = all_sentences[:num_sentences]
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Voice Recording Script\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total sentences: {len(selected_sentences)}\n")
        f.write("#\n")
        f.write("# Instructions:\n")
        f.write("# - Find a quiet room\n")
        f.write("# - Use a good microphone\n")
        f.write("# - Speak naturally in your normal voice\n")
        f.write("# - Take breaks every 20-30 sentences\n")
        f.write("# - Record each sentence as audio_001.wav, audio_002.wav, etc.\n")
        f.write("#\n")
        f.write("# Format below: audio_XXX.wav|sentence text\n")
        f.write("#" + "="*70 + "\n\n")
        
        for i, sentence in enumerate(selected_sentences, 1):
            f.write(f"audio_{i:03d}.wav|{sentence}\n")
    
    print(f"✓ Recording script generated: {output_file}")
    print(f"✓ Total sentences: {len(selected_sentences)}")
    print(f"\nEstimated recording time: {len(selected_sentences) * 8 / 60:.1f} minutes")
    print(f"(assuming ~8 seconds per sentence including pauses)")

# ============================================================================
# PART 2: Recording Helper Script
# ============================================================================

def create_recording_helper(script_file="recording_script.txt"):
    """
    Create a helper script that displays sentences one by one for easy recording.
    """
    
    helper_code = '''"""
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
    print("\\nInstructions:")
    print("- Press ENTER when ready to see the next sentence")
    print("- Start recording when you see the sentence")
    print("- Stop recording after speaking")
    print("- Save as the filename shown")
    print("- Take breaks every 20-30 recordings")
    print("=" * 80)
    
    input("\\nPress ENTER to start...")
    
    for i, (filename, text) in enumerate(sentences, 1):
        print(f"\\n{'='*80}")
        print(f"Recording {i}/{len(sentences)}")
        print(f"{'='*80}")
        print(f"\\nFilename: {filename}")
        print(f"\\n>>> {text}")
        print(f"\\n{'='*80}")
        
        # Suggest breaks
        if i % 25 == 0 and i < len(sentences):
            print(f"\\n*** SUGGESTION: Take a short break! You've done {i} recordings. ***")
            input("Press ENTER when ready to continue...")
        else:
            input("Press ENTER for next sentence...")
    
    print(f"\\n{'='*80}")
    print("🎉 ALL DONE! You've completed all recordings!")
    print(f"{'='*80}")
    print(f"\\nTotal recordings: {len(sentences)}")
    print("\\nNext steps:")
    print("1. Check that all audio files are properly saved")
    print("2. Listen to a few random recordings to verify quality")
    print("3. Run the training script with your data!")

if __name__ == "__main__":
    display_recording_prompts()
'''
    
    with open("recording_helper.py", 'w', encoding='utf-8') as f:
        f.write(helper_code)
    
    print(f"✓ Recording helper created: recording_helper.py")
    print(f"\nTo use: python recording_helper.py")

# ============================================================================
# PART 3: Prepare Final Metadata for Training
# ============================================================================

def prepare_final_metadata(
    script_file="recording_script.txt",
    audio_folder="my_voice_data",
    output_file="my_voice_data/metadata.txt"
):
    """
    Copy the recording script to the proper metadata.txt format,
    checking which audio files actually exist.
    """
    
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    
    valid_entries = []
    missing_files = []
    
    # Read recording script
    with open(script_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('|')
                if len(parts) == 2:
                    filename, text = parts
                    audio_path = os.path.join(audio_folder, filename)
                    
                    if os.path.exists(audio_path):
                        valid_entries.append(line.strip())
                    else:
                        missing_files.append(filename)
    
    # Write valid entries to metadata.txt
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in valid_entries:
            f.write(entry + '\n')
    
    print(f"\\n{'='*80}")
    print("METADATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Valid recordings: {len(valid_entries)}")
    
    if missing_files:
        print(f"⚠ Missing audio files: {len(missing_files)}")
        print(f"\\nFirst 5 missing files:")
        for f in missing_files[:5]:
            print(f"  - {f}")
    
    print(f"\\n✓ Metadata file ready: {output_file}")
    print(f"\\nYou can now run the training script!")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("VOICE CLONING DATASET PREPARATION")
    print("=" * 80)
    
    # Step 1: Generate recording script
    print("\\nStep 1: Generating recording script...")
    generate_metadata_file(
        output_file="recording_script.txt",
        num_sentences=200  # Adjust this number as needed
    )
    
    # Step 2: Create recording helper
    print("\\nStep 2: Creating recording helper...")
    create_recording_helper()
    
    print("\\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\\n1. Review recording_script.txt to see all sentences")
    print("\\n2. Set up your recording equipment:")
    print("   - Find a quiet room")
    print("   - Position your microphone")
    print("   - Test recording quality")
    print("\\n3. Run: python recording_helper.py")
    print("   - Follow the prompts")
    print("   - Record each sentence")
    print("   - Save as my_voice_data/audio_001.wav, etc.")
    print("\\n4. After recording, run this script with --prepare flag:")
    print("   - It will check your recordings and create metadata.txt")
    print("\\n5. Then run the training script!")
    
    print("\\n" + "=" * 80)
    
    # Optional: Prepare metadata if recordings exist
    if input("\\nDo you want to check existing recordings now? (y/n): ").lower() == 'y':
        prepare_final_metadata()