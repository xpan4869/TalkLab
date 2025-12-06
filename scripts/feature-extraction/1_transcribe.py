# 1_transcribe.py
# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: Dec 5, 2025

# Description:
#   Extracts audio from an MP4 file and runs:
#     1) WhisperX ASR
#     2) Forced alignment
#     3) Pyannote-based speaker diarization (via WhisperX)
#   Produces a speaker-labeled, segment-level transcript as CSV.

# Reference: https://github.com/m-bain/whisperX


import whisperx
import gc
import torch
import os
import csv
from whisperx.diarize import DiarizationPipeline

# =========================================================================
#  UPDATED FIX FOR PYTORCH 2.6+ (AGGRESSIVE)
# =========================================================================
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # FORCE weights_only=False, even if the internal library tries to set it to True
    kwargs['weights_only'] = False 
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
# =========================================================================

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/yolandapan/TalkLab/scripts/feature-extraction')
_THISDIR = os.getcwd()
VIDEO_FILE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/db73acf1-9a4d-405e-8b7d-fa44529f1e81_copy.mp4'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/processed'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") # Required for Pyannote Diarization
# HF_TOKEN = None # if 'huggingface login' 

if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

WHISPER_MODEL_SIZE = "large-v3"
    
BATCH_SIZE = 16
MIN_SPEAKERS = 2
MAX_SPEAKERS = 2


# ------------------ Define Functions ------------------ #
def run_asr(audio_file):
    """Transcribe with original whisper (batched)"""
    print("Loading transcribe model...")
    model = whisperx.load_model(WHISPER_MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=BATCH_SIZE) # before alignment
    language_code = result["language"]

    # Clean up model to save gpu space
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return audio, result, language_code 

def run_alignment(segments, audio, language_code: str):
    """
    Force-align ASR segments to improve timestamps.
    """
    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=DEVICE
    )

    aligned = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    ) # after alignment

    del model_a
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return aligned 

def run_diarization(audio, hf_token: str):
    """
    Run speaker diarization to assign speaker labels using WhisperX Pyannote pipeline.
    """
    if not hf_token:
        raise ValueError(
            "HUGGINGFACE_TOKEN not set. Please set HUGGINGFACE_TOKEN env var "
            "or pass a valid token to the pipeline."
        )

    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token,
        device=DEVICE,
    )

    diarize_segments = diarize_model(
        audio,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS,
    )

    return diarize_segments


def export_segment_level_transcript(result_with_speakers, save_dir: str) -> str:
    """
    Extract segment-level speaker info (speaker, start, end, text)
    and save as a CSV file.
    """
    segments_out = []
    for seg in result_with_speakers.get("segments", []):
        segments_out.append({
            "speaker": seg.get("speaker", "UNKNOWN_SPEAKER"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", "").strip(),
        })

    out_path = os.path.join(save_dir, "speaker_segments.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker", "start", "end", "text"]
        )
        writer.writeheader()
        for row in segments_out:
            writer.writerow(row)

    print(f"Saved {len(segments_out)} speaker-labeled segments to CSV:")
    print(f"{out_path}")

    return out_path


# ------------------ Main ------------------ #
if __name__ == "__main__":    
    print(f"DEVICE     = {DEVICE}")
    print(f"VIDEO_FILE = {VIDEO_FILE_PATH}")

    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"Error: Video file not found at '{VIDEO_FILE_PATH}'.")
    elif not HF_TOKEN:
        print("Error: HUGGINGFACE_TOKEN is not set in the environment.")
    else:
        if DEVICE == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA reported but not actually available. Falling back to CPU.")

        audio, asr_result, language_code = run_asr(VIDEO_FILE_PATH)
        aligned_result = run_alignment(asr_result["segments"], audio, language_code)
        diarize_segments = run_diarization(audio, HF_TOKEN)
        print("Assigning speaker labels to aligned segments...")
        result_with_speakers = whisperx.assign_word_speakers(
            diarize_segments,
            aligned_result,
        )
        
        out_csv = export_segment_level_transcript(result_with_speakers, SAVE_PATH)

        print("Done.")
