"""Generate all projections from Epic 3D with multi-threading
"""
import os
import sys
from time import time, sleep
from glob import glob
import pickle
import argparse
import threading
import logging
import queue

from natsort import natsorted

from read_3d_data import (read_corpus,
                          read_intrinsic_extrinsic,
                          project_frame)

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
BUF_SIZE = 10
q = queue.Queue(BUF_SIZE)


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
        results = pickle.load(open(result_path, 'rb'))
        report = pickle.load(open(report_path, 'rb'))
        return results, report

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
    report['time_proj_avg'] = proj_time / (vinfo.nframes - len(report['broken_frames']))

    # Save results ------------------------------------------------------------
    pickle.dump(report, open(report_path, 'wb'))
    pickle.dump(results, open(result_path, 'wb'))

    return results, report


class ProducerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ProducerThread, self).__init__()
        self.target = target
        self.name = name

        self.vid_list = [x for x in kwargs['vid_list']]

    def run(self):
        while self.vid_list != []:
            if not q.full():
                vid_path = self.vid_list.pop(0)
                q.put(vid_path)
                logging.debug('Putting ' + str(vid_path)
                              + ' : ' + str(q.qsize()) + ' items in queue')
                sleep(1)
        logging.debug('Ending')


class ConsumerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ConsumerThread, self).__init__()
        self.target = target
        self.name = name

        self.vid_list = [x for x in kwargs['vid_list']]
        self.report_dir = kwargs['report_dir']
        self.result_dir = kwargs['result_dir']

    def run(self):
        # while True:
        while self.vid_list != []:
            if not q.empty():
                # Get item from queue
                vid_path = q.get()
                self.vid_list.remove(vid_path)
                logging.debug('Getting ' + str(vid_path)
                              + ' : ' + str(q.qsize()) + ' items in queue')

                # Process item
                vid_id = os.path.basename(vid_path)
                result_path = os.path.join(self.result_dir, vid_id+'.pkl')
                report_path = os.path.join(self.report_dir, vid_id+'.pkl')
                project_vid(vid_path, result_path, report_path)

                sleep(1)
        logging.debug('Ending')


def main(args):
    # Retrieve list of all videos
    vid_list = [item for item in glob(os.path.join(args.data_root, '*'))
                if os.path.isdir(item)]
    vid_list = natsorted(vid_list)

    # Create producer and consumer for multi threading
    producer = ProducerThread(name='producer', kwargs={'vid_list': vid_list})
    consumer = ConsumerThread(name='consumer', kwargs={'vid_list': vid_list,
                                                       'result_dir': args.result_dir,
                                                       'report_dir': args.report_dir})

    producer.start()
    sleep(2)
    consumer.start()
    sleep(2)

    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
