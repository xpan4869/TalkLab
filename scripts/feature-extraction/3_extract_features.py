# 3_extract_features.py
# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: Dec 6, 2025

# Description: 
#   Enriches conversational turns with multimodal features:
#   1. Text Feature: Sentiment Valence (VADER) - Measures emotional content (-1 to +1).
#   2. Audio Feature: Pitch Variability (F0 Std Dev) - Measures vocal engagement/prosody.

import pandas as pd
import numpy as np
import librosa
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# ------------------ Hardcoded parameters ------------------ #
TEXT_FILE = "/Users/yolandapan/TalkLab/data/processed/conversational_turns.csv"
AUDIO_FILE = "/Users/yolandapan/TalkLab/data/db73acf1-9a4d-405e-8b7d-fa44529f1e81_copy.mp4"
OUTPUT_CSV = "/Users/yolandapan/TalkLab/data/processed/turns_with_features.csv"

sia = SentimentIntensityAnalyzer()

# ------------------ Define Functions ------------------ #
def get_sentiment(text: str) -> float:
    """
    Returns compound sentiment score: 
    -1.0 (Most Negative) to +1.0 (Most Positive).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    return sia.polarity_scores(text)['compound']

def get_pitch_variability(y_segment, sr):
    """
    Extracts Pitch Variability (Standard Deviation of F0).
    Uses Librosa's probabilistic YIN (pYIN) algorithm.
    """
    if len(y_segment) < 512: 
        return 0.0

    f0, _, _ = librosa.pyin(
        y_segment, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C5'),
        sr=sr
    )
    
    # Filter out NaNs (unvoiced parts like silence or breath)
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) == 0:
        return 0.0

    return np.std(f0_clean)

# ------------------ Main ------------------ #
def main():
    if not os.path.exists(TEXT_FILE):
        print(f"Input CSV not found: {TEXT_FILE}")
        return
    
    if not os.path.exists(AUDIO_FILE):
        print(f"Audio file not found: {AUDIO_FILE}")
        return

    df = pd.read_csv(TEXT_FILE)
    y, sr = librosa.load(AUDIO_FILE, sr=16000)
    
    sentiments = []
    pitch_stds = []

    for i, row in df.iterrows():
# Text Analysis
        text = str(row['text'])
        sentiments.append(get_sentiment(text))        
        
        # Audio Analysis
        start_sample = max(0, int(row['start'] * sr))
        end_sample = min(len(y), int(row['end'] * sr))
        
        y_turn = y[start_sample:end_sample]
        std_p = get_pitch_variability(y_turn, sr)
        pitch_stds.append(std_p)

        if i % 20 == 0:
            print(f"   Processed {i}/{len(df)}...", end='\r')

    df['sentiment'] = sentiments
    df['pitch_std'] = pitch_stds  
    
    # Fill NaNs (silence) with 0
    df['pitch_std'] = df['pitch_std'].fillna(0)

    # Rounding
    df['sentiment'] = df['sentiment'].round(3)
    df['pitch_std'] = df['pitch_std'].round(3)

    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n Features Extracted. Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()