# Spatiotemperal Sparse Sensing for Ego-Action Recognition

## Data preprocessing

- Download EPIC-KITCHENS to `data` folder or create a symlinks. Note that we do not need `object_detection_images` for this project. You should have something like this:
    ```
    sparse_sensing_actreg/
    └── data/
        └── EPIC-KITCHENS-2018/
            ├── frames_rgb_flow/
            └── videos/
    ```
    - `frames_rgb_flow`: contains the RGB and optical flow frames as tar files
    - `videos`: contains the videos with audio. We will extract audio from this
- Run `scripts/preprocess/untar_data.sh` to extract the frames from `frames_rgb_flow/` to `frames_untar/`. It is preferred to keep the extracted frames separated for easier transferring.
- Run `scripts/preprocess/preprocess_epic.sh` to preprocess the dataset. This will:
    1. Restructure the dataset by creating symlinks to `frames_untar/` and put them in `frames_restruct/`
    2. Extract audio from `videos/` and place in `audio/`
    3. Create wave dictionary from the extracted audio
    (The 3 steps can be run independently.)
