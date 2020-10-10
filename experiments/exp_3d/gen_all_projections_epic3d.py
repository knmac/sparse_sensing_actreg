"""Generate all projections from Epic 3D with multi-threading
"""
import os
import sys
from time import time
from glob import glob
import pickle
import argparse
from threading import Thread
from queue import Queue

from natsort import natsorted

from read_3d_data import (read_corpus,
                          read_intrinsic_extrinsic,
                          project_frame)


def parse_args():
    """Parser input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', type=str,
        default='/home/knmac/datasets/epic3D/epic18_completed',
        help='Directory containing data',
    )
    parser.add_argument(
        '--result_dir', type=str,
        default='/home/knmac/datasets/epic3D/projections',
        help='Directory to store projections (as pickle format)',
    )
    parser.add_argument(
        '--report_dir', type=str,
        default='/home/knmac/datasets/epic3D/reports',
        help='Directory to store reports (as pickle format)',
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of workers to process data',
    )

    args = parser.parse_args()
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.isdir(args.report_dir):
        os.makedirs(args.report_dir)
    return args


def project_vid(vid_path, result_path, report_path):
    """Project frames in a video

    Args:
        vid_path: path to the directory containing data of a video
        result_path: path to save the projection results
        report_path: path to save the reports

    Returns:
        results: dictionary of projection results for each frame
        report: dictionary of time_corpus, time_intr_extr, broken_frames,
            time_proj_avg, and time_proj_total for each frame
    """
    # Skip if files exist
    if os.path.isfile(result_path) and os.path.isfile(report_path):
        print('    File exists -> Skipped')
        return
        # results = pickle.load(open(result_path, 'rb'))
        # report = pickle.load(open(report_path, 'rb'))
        # return results, report

    results = {}
    report = {}

    # Read corpus of 3D points in a video -------------------------------------
    st = time()
    corpus_info, vcorpus_cid_lcid_lfid = read_corpus(vid_path)
    report['time_corpus'] = time() - st

    # Read intrinsic and extrinsic parameters ---------------------------------
    st = time()
    vinfo = read_intrinsic_extrinsic(vid_path)
    report['time_intr_extr'] = time() - st

    # Project frames ----------------------------------------------------------
    report['broken_frames'] = []

    st = time()
    for frame_id in range(vinfo.nframes):
        # Project the frame
        projected, depths, colors, points3d = project_frame(
            frame_id, vinfo, corpus_info, vcorpus_cid_lcid_lfid,
            unique_points=True)

        # Collect results
        results[frame_id] = {
            'projected': projected,
            'depths': depths,
            'colors': colors,
            'points3d': points3d,
        }

        if projected is None:
            report['broken_frames'].append(frame_id)

    proj_time = time() - st
    report['time_proj_total'] = proj_time
    if vinfo.nframes == len(report['broken_frames']):
        print('    All frames broken: {}'.format(os.path.basename(vid_path)))
        return

    report['time_proj_avg'] = proj_time / (vinfo.nframes - len(report['broken_frames']))

    # Save results ------------------------------------------------------------
    pickle.dump(report, open(report_path, 'wb'))
    pickle.dump(results, open(result_path, 'wb'))

    # return results, report


class MyWorker(Thread):
    def __init__(self, queue, result_dir, report_dir):
        Thread.__init__(self)
        self.queue = queue
        self.result_dir = result_dir
        self.report_dir = report_dir

    def run(self):
        while True:
            vid_path = self.queue.get()
            try:
                vid_id = os.path.basename(vid_path)
                result_path = os.path.join(self.result_dir, vid_id+'.pkl')
                report_path = os.path.join(self.report_dir, vid_id+'.pkl')
                print('Processing {}. Queue size = {}'.format(vid_id, self.queue.qsize()))
                project_vid(vid_path, result_path, report_path)
            finally:
                self.queue.task_done()
                print('--> Finished {}...'.format(vid_id))


def main(args):
    # Retrieve list of all videos
    vid_list = [item for item in glob(os.path.join(args.data_root, '*'))
                if os.path.isdir(item)]
    vid_list = natsorted(vid_list)

    queue = Queue()
    for x in range(args.num_workers):
        worker = MyWorker(queue, args.result_dir, args.report_dir)
        worker.daemon = True
        worker.start()

    for vid_path in vid_list:
        queue.put(vid_path)

    queue.join()

    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
