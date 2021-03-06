"""Clean up invalid videos for RGBDS data
"""
import pandas as pd


STAT = "dataset_splits/EPIC_KITCHENS_2018/fTrackFramesSuccess.txt"
IN_TRAIN = "dataset_splits/EPIC_KITCHENS_2018/EPIC_train_action_labels.pkl"
IN_VAL = "dataset_splits/EPIC_KITCHENS_2018/EPIC_val_action_labels.pkl"

OUT_TRAIN = "dataset_splits/EPIC_KITCHENS_2018/EPIC_train_action_labels_rgbds.pkl"
OUT_VAL = "dataset_splits/EPIC_KITCHENS_2018/EPIC_val_action_labels_rgbds.pkl"


def clean(data_pth, selection):
    data = pd.read_pickle(data_pth)
    data_clean = data[data['video_id'].isin(selection)]
    return data_clean


def main():
    # Parse statistics file
    with open(STAT) as fin:
        content = fin.read().splitlines()
        key_percent = {'P'+line.split()[0]: float(line.split()[-1])
                       for line in content}

    # Select valid videos
    selection = []
    for k in key_percent:
        if key_percent[k] > 30:
            selection.append(k)
        else:
            print(k, key_percent[k])
    print(len(selection), len(key_percent))

    # Clean up
    train_clean = clean(IN_TRAIN, selection)
    train_clean.to_pickle(OUT_TRAIN)

    val_clean = clean(IN_VAL, selection)
    val_clean.to_pickle(OUT_VAL)


if __name__ == '__main__':
    main()
