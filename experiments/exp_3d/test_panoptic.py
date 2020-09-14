import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt


def parse_segment_id(segment_id, segments_info, thing_2_contiguous, stuff_2_contiguous,
                     thing_classes, stuff_classes):
    if segment_id == 0:
        return 0, ''
    info = segments_info[segment_id-1]
    assert info['id'] == segment_id
    if info['isthing']:
        return thing_2_contiguous[info['category_id']], thing_classes[info['category_id']]
    return stuff_2_contiguous[info['category_id']], stuff_classes[info['category_id']]


def main():
    im = cv2.imread('../tmp_extract/frames_full/P01_08/0/0020.jpg')

    # Inference with a panoptic segmentation model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    thing_2_contiguous = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
    stuff_2_contiguous = {v: k for k, v in metadata.stuff_dataset_id_to_contiguous_id.items()}

    # My code
    panoptic_seg = panoptic_seg.cpu().numpy()
    semantic = np.zeros(panoptic_seg.shape, dtype=int)
    for item in segments_info:
        print(item)

    for i in np.unique(panoptic_seg):
        lbl, meaning = parse_segment_id(i, segments_info, thing_2_contiguous, stuff_2_contiguous,
                                        metadata.thing_classes, metadata.stuff_classes)
        semantic[panoptic_seg == i] = lbl
        print(i, '-->', lbl, ':', meaning)

    fig, axes = plt.subplots(2, 9, figsize=(10, 10))
    for i in np.unique(panoptic_seg):
        lbl, meaning = parse_segment_id(i, segments_info, thing_2_contiguous, stuff_2_contiguous,
                                        metadata.thing_classes, metadata.stuff_classes)
        foo = np.copy(panoptic_seg)
        bar = np.copy(semantic)
        foo[foo != i] = 0
        bar[bar != lbl] = 0
        axes[0, i].imshow(foo)
        axes[1, i].imshow(bar)
        axes[1, i].set_xlabel(meaning)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(out.get_image())
    plt.show()


if __name__ == '__main__':
    main()
