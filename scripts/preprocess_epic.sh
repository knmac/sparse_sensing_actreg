# Preprocess EPIC-KITCHENS dataset

# -----------------------------------------------------------------------------
# Symlinks
# -----------------------------------------------------------------------------
echo "Run symlinks..."

# Get the absolute path to frame dir for symlinks
abs_data_path=$(realpath -s "data/EPIC_KITCHENS_2018/frames_rgb_flow")

# Create symlinks
python tools/preprocessing_epic/symlinks.py \
    --data_dir $abs_data_path \
    --symlinks_dir "data/EPIC_KITCHENS_2018/frames_unified"


# -----------------------------------------------------------------------------
# Extract audio
# -----------------------------------------------------------------------------
echo "\nExtracting audio..."

python tools/preprocessing_epic/extract_audio.py \
    --videos_dir "data/EPIC_KITCHENS_2018/videos" \
    --output_dir "data/EPIC_KITCHENS_2018/audio"


# -----------------------------------------------------------------------------
# Create wav dictionary
# -----------------------------------------------------------------------------
echo "\nCreating wav dictionary..."

python tools/preprocessing_epic/wav_to_dict.py \
    --sound_dir "data/EPIC_KITCHENS_2018/audio" \
    --output_dir "data/EPIC_KITCHENS_2018/audio_dict.pkl"
