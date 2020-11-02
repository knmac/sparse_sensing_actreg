"""Parse the tensorboard quickly to find the best prec@1
"""
import sys
import os
import glob
import argparse

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--log_dir', type=str)

    args = parser.parse_args()
    assert os.path.isdir(args.log_dir)
    return args


def get_scalars(log_dir, log_type):
    event_fname = glob.glob(os.path.join(log_dir, log_type, 'event*'))
    if len(event_fname) != 1:
        print('  {} --> len(event_fname)=={}'.format(log_type, len(event_fname)))
        return None

    ttf_guidance = {
        event_accumulator.SCALARS: 0,
    }
    event_fname = event_fname[0]
    ea = event_accumulator.EventAccumulator(event_fname, ttf_guidance)
    ea.Reload()

    scalars = [ea.Scalars(tag) for tag in ea.Tags()['scalars']]
    return scalars


def main(args):
    """Main function"""
    # Find argmax wrt data_prec_top1
    log_type = 'data_prec_top1_validation'
    top1 = get_scalars(args.log_dir, log_type)[0]
    print('n_epochs =', len(top1))
    argmax = np.argmax([item.value for item in top1])

    # Print all data at argmax
    log_types = [
        'data_loss_validation',
        'data_prec_top1_validation',
        'data_prec_top5_validation',
        'data_verb_loss_validation',
        'data_verb_prec_top1_validation',
        'data_verb_prec_top5_validation',
        'data_noun_loss_validation',
        'data_noun_prec_top1_validation',
        'data_noun_prec_top5_validation',
    ]
    for log_type in log_types:
        data = get_scalars(args.log_dir, log_type)
        if data is None:
            continue
        data = data[0]
        print('{} --> {:.03f}'.format(log_type, data[argmax].value))
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
