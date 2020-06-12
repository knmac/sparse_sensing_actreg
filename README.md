# Spatio-Temperal Sparse Sensing for Ego-Action Recognition

## Data preprocessing

- Download EPIC-KITCHENS to `data` folder or create a symlinks. You should have something like this:
```
root/
    data/
        EPIC-KITCHENS-2018/
            frames_rgb_flow/
            videos/
            ...
```
- Run `scripts/untar_data.sh` to extract the frames from `frames_rgb_flow/` to `frames_untar/` (I keep the extracted frames separated for easy transferring)
- Run `scripts/preprocess_epic.sh` to preprocess the dataset. This will
    1. Restructure the dataset by creating symlinks to `frames_untar/` and put them in `frames_restruct/`
    2. Extract audio from `videos/` and place in `audio/`
    3. Create wave dictionary from the extracted audio
(The 3 steps can be run separately.)
