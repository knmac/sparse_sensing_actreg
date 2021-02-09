"""ActivityNet dataset"""
import sys
import os

import torch
import numpy as np
from numpy.random import randint
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from src.datasets.base_dataset import BaseDataset
from src.datasets.video_record import VideoRecord
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class ActivityNetDataset(BaseDataset):
    def __init__(self, mode, list_file, modality=['RGB'], image_tmpl='{:04d}.jpg',
                 visual_path=None, num_segments=3, transform=None,
                 new_length=None, remove_missing=False):
        super(ActivityNetDataset, self).__init__(mode)
        self.name = 'activity_net'

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        if not os.path.isdir(visual_path):
            visual_path = os.path.join(root, visual_path)
        self.visual_path = visual_path

        if list_file[mode] is not None:
            # Use the given lists
            if not os.path.isfile(list_file[mode]):
                list_file[mode] = os.path.join(root, list_file[mode])
            self.list_file = list_file[mode]

        self.modality = modality
        assert modality == ['RGB'], 'Only support RGB for now'
        # TODO: allow other modalities
        self.image_tmpl = image_tmpl['RGB']
        self.transform = transform['RGB']
        self.new_length = new_length['RGB']

        self.mode = mode
        self.num_segments = num_segments
        self.remove_missing = remove_missing

        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        record = self.video_list[index]

        # Check this is a legit video folder using frame 1
        file_name = self.image_tmpl.format(1)
        full_path = os.path.join(self.visual_path, record.path, file_name)

        err_cnt = 0
        while not os.path.exists(full_path):
            err_cnt += 1

            if err_cnt > 3 and not self.remove_missing:
                logger.info('Something wrong with the dataloader to get items')
                logger.info('Check your data path. Exit...')
                raise

            if not self.remove_missing:
                logger.info('Not found: {}'.format(full_path))

            # Try another video
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.visual_path, record.path, file_name)

        # Get the frame indices wrt the current mode
        if self.mode == 'train':
            segment_indices = self._sample_indices(record)
        elif self.mode == 'val':
            segment_indices = self._get_val_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        img, label = self.get(record, segment_indices)
        return {'RGB': img}, label

    def _parse_list(self):
        with open(self.list_file) as f:
            tmp = [x.strip().split(',') for x in f]

        if (self.mode != 'test') or (self.remove_missing):
            tmp = [item for item in tmp if int(item[1]) >= 3]

        self.video_list = [ActivityVideoRecord(item, self.visual_path) for item in tmp]

    def _load_image(self, directory, idx):
        try:
            path = os.path.join(self.visual_path, directory, self.image_tmpl.format(idx))
        except Exception:
            logger.info('Error loading image: %s', path)
            path = os.path.join(self.visual_path, directory, self.image_tmpl.format(1))
        return [Image.open(path).convert('RGB')]

    def _sample_indices(self, record):
        average_duration = record.num_frames // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames, size=self.num_segments))
        else:
            offsets = list(range(record.num_frames)) + \
                [record.num_frames - 1] * (self.num_segments - record.num_frames)
            offsets = np.array(offsets)
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments:
            tick = record.num_frames / float(self.num_segments)
            offsets = [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
        else:
            offsets = list(range(record.num_frames)) + \
                [record.num_frames - 1] * (self.num_segments - record.num_frames)
        offsets = np.array(offsets)
        return offsets + 1

    def _get_test_indices(self, record):
        tick = record.num_frames / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            images.extend(self._load_image(record.path, int(seg_ind)))

        process_data = self.transform(images)
        # TODO: resize to other resolutions can go here
        return process_data, record.label


class ActivityVideoRecord(VideoRecord):
    def __init__(self, row, visual_path):
        self._data = row
        self._labels = torch.tensor([-1, -1, -1])
        labels = sorted(list(set([int(x) for x in self._data[2:]])))
        for i, l in enumerate(labels):
            self._labels[i] = l

        self._data[1] = int(self._data[1])
        vid_dir = os.path.join(visual_path, self._data[0])
        if os.path.isdir(vid_dir):
            self._data[1] = len(os.listdir(vid_dir))

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return self._data[1]

    @property
    def label(self, retrieve_single=True):
        if retrieve_single:
            return self._labels[0]
        return self._labels
