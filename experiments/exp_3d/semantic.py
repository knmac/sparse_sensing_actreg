"""Retrieve semantic using majority voting on panoptic segmentation
"""
import os

import numpy as np
import torch
import cv2
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt

from read_3d_data import (read_corpus,
                          read_intrinsic_extrinsic,
                          match_point_to_frame,
                          project_and_distort)


def get_panop_keyframes(keyframe_ids, frame_path, id_offset=0):
    """Get panoptic segmentation of keyframes

    Args:
        keyframe_ids: list of keyframe IDs
        frame_path: path to the frames
        id_offset: offset of the first frame index

    Return:
        panoptic_seg_dict: dictionary of panoptic segmentation, each
            corresponds to a key frame. Each panoptic_seg is a 2D array
            where the values are segment IDs. The meaning of the segments can
            be looked up using `segments_info`
        segments_info_dict: dictionary of segments info, each corresponds to
            a key frame. Each item contains the information of a panoptic
            segmentation map
        metadata: metadata of the dataset wrt the config
    """
    # Prepare segmentation model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml')
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Go through the list of keyframe and retrieve panoptic segmentation
    panoptic_seg_dict, segments_info_dict = {}, {}
    for kf_id in tqdm(keyframe_ids):
        fname = os.path.join(frame_path,
                             '{:04d}.jpg'.format(kf_id+id_offset))
        img = cv2.imread(fname)
        panoptic_seg, segments_info = predictor(img)['panoptic_seg']

        panoptic_seg_dict[kf_id] = panoptic_seg.to('cpu').numpy()
        segments_info_dict[kf_id] = segments_info

    return panoptic_seg_dict, segments_info_dict, metadata


def parse_segment_id(segment_id, segments_info, thing_2_contiguous,
                     stuff_2_contiguous, thing_classes, stuff_classes):
    """Parse the information wrt to a segment ID

    If the segment_id == 0, return dummy values as 0 means non-segmented area

    Args:
        segment_id: ID of the segment to retrieve information
        segments_info: information of the whole panoptic segmentation map. This
            is a list where each item is the information of a segment
        thing_2_contiguous: mapping from `thing` to contiguous labels. The
            contiguous ones contains labels from both `thing` and `stuff`
            categories in a contiguous manner
        stuff_2_contiguous: mapping from `stuff` to contiguous labels
        thing_classes: meaning of labels of `thing` categories
        stuff_classes: meaning of labels of `stuff` categories

    Return:
        semantic: contiguous label of the segment_id
        meaning: meaning of the label
        score: score of the segment_id. Only valid if the segment_id is from
            `thing` category. If it is from `stuff` category, return 1.0
    """
    if segment_id == 0:
        return 0, '', 0.0
    info = segments_info[segment_id-1]
    assert info['id'] == segment_id
    if info['isthing']:
        return (thing_2_contiguous[info['category_id']],
                thing_classes[info['category_id']],
                info['score'])
    return (stuff_2_contiguous[info['category_id']],
            stuff_classes[info['category_id']],
            1.0)


def vote(semantic_candidates, meaning_candidates, score_candidates, scheme):
    """Vote for the most relevant semantic from a list of candidates

    Args:
        semantic_candidates: list of candidate semantic labels
        meaning_candidates: list of meaning of the candidate semantic labels
        score_candidates: list of scores of the candidate semantic labels
        scheme: `majority` or `weighted` only
            - `majority`: vote by the majority of appearance
            - `weighted`: vote by the score of the candidate

    Return:
        semantic: voted semantic label
        meaning: the corresponding meaning of the voted semantic label
    """
    # Remove semantic with zero label because it means non-detected results
    semantic_candidates = np.array(semantic_candidates)
    semantic_candidates_filter = semantic_candidates[semantic_candidates != 0]

    # Vote if there are still candidates
    if semantic_candidates_filter.size != 0:
        # Majority voting
        if scheme == 'majority':
            semantic = np.bincount(semantic_candidates_filter).argmax()
            tmp = np.where(semantic_candidates == semantic)[0][0]
            meaning = meaning_candidates[tmp]
        # Weighted voting
        elif scheme == 'weighted':
            unique_semantic = np.unique(semantic_candidates_filter)
            _good_score = -1000
            _good_semantic = -1
            score_candidates = np.array(score_candidates)
            for ii in unique_semantic:
                _idx = np.where(semantic_candidates == ii)
                _score = score_candidates[_idx].sum() / len(semantic_candidates_filter)
                if _score > _good_score:
                    _good_score = _score
                    _good_semantic = ii
            semantic = _good_semantic
            tmp = np.where(semantic_candidates == semantic)[0][0]
            meaning = meaning_candidates[tmp]
        else:
            raise NotImplementedError
    else:
        semantic, meaning = 0, ''
    return semantic, meaning


def main():
    path = '/home/knmac/Documents/Dropbox/SparseSensing/3d_projection/P01_08'
    frame_path = '/home/knmac/projects/tmp_extract/frames_full/P01_08/0'
    panoptic_path = './cache/panoptic_dict_P01_08.data'
    semantic_path = './cache/semantic.data'

    # Read the 3D points
    print('Read corpus...')
    CorpusInfo, vCorpus_cid_Lcid_Lfid = read_corpus(path)
    point_frame_lookup = match_point_to_frame(CorpusInfo, vCorpus_cid_Lcid_Lfid)

    # Read camera parameters
    print('Read camera...')
    vInfo = read_intrinsic_extrinsic(path, stopF=5931)

    # Get panoptic segmentation of keyframes
    print('Get panoptic segmentation...')
    if os.path.isfile(panoptic_path):
        data = torch.load(panoptic_path)
        panoptic_seg_dict = data['panoptic_seg_dict']
        segments_info_dict = data['segments_info_dict']
        metadata = data['metadata']
    else:
        (panoptic_seg_dict,
         segments_info_dict,
         metadata) = get_panop_keyframes(vCorpus_cid_Lcid_Lfid[:, 2],
                                         frame_path, id_offset=1)
        data = {'panoptic_seg_dict': panoptic_seg_dict,
                'segments_info_dict': segments_info_dict,
                'metadata': metadata}
        torch.save(data, panoptic_path)

    thing_2_contiguous = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
    stuff_2_contiguous = {v: k for k, v in metadata.stuff_dataset_id_to_contiguous_id.items()}

    # Project and find semantic label of a 3D point
    print('Compute point semantic...')
    if os.path.isfile(semantic_path):
        data = torch.load(semantic_path)
        semantic_lst = data['semantic_lst']
        meaning_lst = data['meaning_lst']
    else:
        # Zeros means no semantic label
        semantic_lst = np.zeros(CorpusInfo.n3dPoints, dtype=np.int32)
        meaning_lst = [None for _ in range(CorpusInfo.n3dPoints)]

        # For each point, pt_id is the id of a point
        for pt_id in tqdm(range(CorpusInfo.n3dPoints)):
            pt_3d = CorpusInfo.xyz[pt_id]  # 3D pos of a point

            # List of candidate info wrt different frames
            semantic_candidates = []
            meaning_candidates = []
            score_candidates = []

            # For each frame that contains pt_id
            for kf_id in point_frame_lookup[pt_id]:
                # Camera info of that frame
                cam_info = vInfo.VideoInfo[kf_id]
                if cam_info.P is None:
                    continue

                # 3D -> 2D projection
                pt_2d = project_and_distort(pt_3d, cam_info.P, cam_info.K, cam_info.distortion)
                x, y = pt_2d.astype(int)

                # Get the segment_id wrt the 2d location
                segment_id = panoptic_seg_dict[kf_id][y, x]

                # Look up the semantic wrt the segment_id
                (semantic,
                 meaning,
                 score) = parse_segment_id(segment_id, segments_info_dict[kf_id],
                                           thing_2_contiguous, stuff_2_contiguous,
                                           metadata.thing_classes,
                                           metadata.stuff_classes)
                semantic_candidates.append(semantic)
                meaning_candidates.append(meaning)
                score_candidates.append(score)

            # Find the good semantic from all candidates
            if semantic_candidates != []:
                (semantic_lst[pt_id],
                 meaning_lst[pt_id]) = vote(semantic_candidates,
                                            meaning_candidates,
                                            score_candidates,
                                            scheme='majority')
        data = {'semantic_lst': semantic_lst,
                'meaning_lst': meaning_lst}
        torch.save(data, semantic_path)

    # Remove object with too few points
    counters = np.bincount(semantic_lst)
    for i in range(len(counters)):
        if counters[i] < 100:
            semantic_lst[semantic_lst == i] = 0

    # Visualize
    unique_semantic = np.unique(semantic_lst)
    # print(unique_semantic)
    visualizing_scheme = 'all'
    # visualizing_scheme = 'by_class'

    if visualizing_scheme == 'all':
        # Generate unique color for each semantic class
        cmap = plt.cm.get_cmap('hsv', len(unique_semantic))
        colors = []
        for i in range(CorpusInfo.n3dPoints):
            color = cmap(np.where(unique_semantic == semantic_lst[i])[0][0])
            colors.append(color)
        colors = np.array(colors)

        # Plot as a whole
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        for semantic in unique_semantic:
            if semantic == 0:
                continue
            idx = np.where(semantic_lst == semantic)
            ax.scatter(CorpusInfo.xyz[idx, 0], CorpusInfo.xyz[idx, 1], CorpusInfo.xyz[idx, 2],
                       s=1, alpha=0.3, c=colors[idx], label=np.array(meaning_lst)[idx][0])
        ax.legend()
        plt.show()
    elif visualizing_scheme == 'by_class':
        # Create a different plot for each semantic class
        for semantic in unique_semantic:
            if semantic == 0:
                continue
            idx = np.where(semantic_lst == semantic)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(CorpusInfo.xyz[idx, 0], CorpusInfo.xyz[idx, 1], CorpusInfo.xyz[idx, 2],
                       s=1, c=CorpusInfo.rgb[idx].astype(np.float32) / 255)
            plt.title(meaning_lst[idx[0][0]])
        plt.show()


if __name__ == '__main__':
    main()
