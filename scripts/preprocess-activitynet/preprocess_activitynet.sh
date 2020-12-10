#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Extract audio
# -----------------------------------------------------------------------------
echo "\nExtracting audio..."

python tools/preprocessing_activitynet/extract_audio.py \
    --videos_dir "data/ActivityNet/ActivityNet-v1.3" \
    --output_dir "data/ActivityNet/audio"


# -----------------------------------------------------------------------------
# Create wav dictionary
# -----------------------------------------------------------------------------
echo "\nCreating wav dictionary..."

python tools/preprocessing_epic/wav_to_dict.py \
    --sound_dir "data/ActivityNet/audio" \
    --output_dir "data/ActivityNet/audio_dict.pkl"
