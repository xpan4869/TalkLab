# 2_turns.py
# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: Dec 6, 2025

# Description: 
#   Transform fragmented WhisperX speech segments into psychologically meaningful conversational turns.
#   It merges same-speaker segments based on pauses 
#   and filters out backchannels (listener feedback) to preserve floor-holding.

# Reference: Cooney, G., Reece, A. NaturalTurn: 
# a method to segment speech into psychologically meaningful conversational turns. 
# Sci Rep 15, 39155 (2025). https://doi.org/10.1038/s41598-025-24381-1


import pandas as pd
import os
import re
from typing import List

# ------------------ Hardcoded parameters ------------------ #
# Update these paths if necessary
INPUT_CSV = "/Users/yolandapan/TalkLab/data/processed/speaker_segments.csv"
OUTPUT_CSV = "/Users/yolandapan/TalkLab/data/processed/conversational_turns.csv"

MAX_PAUSE_THRESHOLD = 1.5
MAX_BC_LENGTH = 3

BACKCHANNEL_CUES = {
    "a", "ah", "alright", "awesome", "cool", "dope", "e", "exactly", 
    "god", "gotcha", "huh", "hmm", "mhm", "mm", "mmm", "nice", 
    "oh", "okay", "really", "right", "sick", "sucks", "sure", 
    "uh", "um", "wow", "yeah", "yep", "yes", "yup"
}

NOT_BACKCHANNEL_CUES = {
    "and", "but", "i", "i'm", "it", "it's", "like", "so", 
    "that", "that's", "we", "we're", "well", "you", "you're"
}

# ------------------ Define Functions ------------------ #
def clean_text_to_words(text: str) -> List[str]:
    """
    Clean text (lowercase, remove punctuation) and tokenize.
    """
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

def is_backchannel(text: str) -> bool:
    """
    Determine if a segment is Secondary Speech (Backchannel).
    Strictly follows the NaturalTurn logic.
    """
    words = clean_text_to_words(text)
    
    if any(w in NOT_BACKCHANNEL_CUES for w in words):
        return False

    if len(words) > MAX_BC_LENGTH:
        return False

    bc_count = sum(1 for w in words if w in BACKCHANNEL_CUES)
    
    if len(words) > 0 and (bc_count / len(words)) >= 0.5:
        return True
    
    return False

# ------------------ Main ------------------ #
def main():
    # Load Data
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    print(f"Reading segments from: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)
    df = df.sort_values(by="start").reset_index(drop=True)

    turns = []
    current_turn = {
        "speaker": df.iloc[0]["speaker"],
        "start": df.iloc[0]["start"],
        "end": df.iloc[0]["end"],
        "text": str(df.iloc[0]["text"]).strip(),
        "bc_count": 0,
        "secondary_speech": [] # Stores ignored backchannel text and timestamps
    }

    # Loop through segments
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        gap = row["start"] - current_turn["end"]
        text = str(row["text"]).strip()

        # --- Scenario 1: Same Speaker ---
        if row["speaker"] == current_turn["speaker"]:
            if gap < MAX_PAUSE_THRESHOLD:
                # Merge: Collapse short pause
                current_turn["end"] = row["end"]
                current_turn["text"] += " " + text
            else:
                # Pause too long: Finalize current turn, start new one
                turns.append(current_turn)
                current_turn = {
                    "speaker": row["speaker"],
                    "start": row["start"],
                    "end": row["end"],
                    "text": text,
                    "bc_count": 0,
                    "secondary_speech": []
                }
        
        # --- Scenario 2: Different Speaker (Potential Turn Switch) ---
        else:
            # Check if this interruption is just a backchannel
            if is_backchannel(text):
                # Current speaker holds the floor.
                current_turn["bc_count"] += 1
                current_turn["secondary_speech"].append(
                    f"{row['speaker']}: [{row['start']:.2f}-{row['end']:.2f}]: {text}"
                )
            else:
                # Switch the floor.
                turns.append(current_turn)
                current_turn = {
                    "speaker": row["speaker"],
                    "start": row["start"],
                    "end": row["end"],
                    "text": text,
                    "bc_count": 0,
                    "secondary_speech": []
                }

    turns.append(current_turn)
    turns_df = pd.DataFrame(turns)
    
    # Format secondary_speech list as string
    turns_df['secondary_speech'] = turns_df['secondary_speech'].apply(lambda x: "; ".join(x))

    # Save
    turns_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()