# Get the real path to data root
real_data_path=$(realpath -s "data/EPIC_KITCHENS_2018/frames_rgb_flow")

# Create symlinks
python tools/preprocessing_epic/symlinks.py \
    --data_dir $real_data_path \
    --symlinks_dir "data/EPIC_KITCHENS_2018/frames_unified"
