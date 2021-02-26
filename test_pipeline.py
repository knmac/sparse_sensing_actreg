"""Run the test on full pipeline with both time and space sampler
"""
import os
import json

import numpy as np
import torch
from tqdm import tqdm

import src.utils.logging as logging
from src.utils.metrics import AverageMeter, accuracy, multitask_accuracy

logger = logging.get_logger(__name__)


def test(model, device, test_loader, args, has_groundtruth):
    # Switch model to eval mode
    model.eval()
    model.to(device)

    # Test
    with torch.no_grad():
        if has_groundtruth:
            logger.info('Testing on val...')
            results = test_with_gt(model, device, test_loader)
            torch.save(results, os.path.join(args.logdir, 'results'))
        else:
            for split, loader in test_loader.items():
                logger.info('Testing on split: {}...'.format(split))
                results, extra_results = test_without_gt(model, device, loader)

                with open(os.path.join(args.logdir, split+'.json'), 'w') as outfile:
                    json.dump(results, outfile)
                torch.save(extra_results, os.path.join(args.logdir, split+'.extra'))


def test_with_gt(model, device, test_loader):
    """Test on the validation set with groundtruth labels
    """
    model_name = type(model).__name__
    assert model_name in ['Pipeline6', 'Pipeline8']
    assert test_loader.dataset.name == 'epic_kitchens', \
        'Unsupported dataset: {}'.format(test_loader.dataset.dataset_name)

    # Prepare metrics
    top1 = AverageMeter()
    top5 = AverageMeter()
    verb_top1 = AverageMeter()
    verb_top5 = AverageMeter()
    noun_top1 = AverageMeter()
    noun_top5 = AverageMeter()
    all_skip, all_time, all_ssim = [], [], []
    all_output = []

    # Test
    for i, (sample, target) in enumerate(test_loader):
        # Forward
        sample = {k: v.to(device) for k, v in sample.items()}
        target = {k: v.to(device) for k, v in target.items()}
        if model_name == 'Pipeline6':
            output, extra_output = model(sample)
        elif model_name == 'Pipeline8':
            output, _, _, extra_output = model(sample, get_extra=True)

        all_skip.append(extra_output['skip'])
        all_time.append(extra_output['time'])
        if isinstance(extra_output['ssim'], np.ndarray):
            all_ssim.append(extra_output['ssim'])
        else:
            all_ssim.append(extra_output['ssim'].cpu().numpy())
        all_output.append(output)

        # Compute metrics
        batch_size = sample[model.modality[0]].size(0)
        verb_output = output[0]
        noun_output = output[1]

        verb_prec1, verb_prec5 = accuracy(verb_output, target['verb'], topk=(1, 5))
        verb_top1.update(verb_prec1, batch_size)
        verb_top5.update(verb_prec5, batch_size)

        noun_prec1, noun_prec5 = accuracy(noun_output, target['noun'], topk=(1, 5))
        noun_top1.update(noun_prec1, batch_size)
        noun_top5.update(noun_prec5, batch_size)

        prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                          (target['verb'], target['noun']),
                                          topk=(1, 5))
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        # Print intermediate results
        if (i % 100 == 0) and (i != 0):
            msg = '[{}/{}]\n'.format(i, len(test_loader))
            msg += '  Prec@1 {:.3f}, Prec@5 {:.3f}\n'.format(top1.avg, top5.avg)
            msg += '  Verb Prec@1 {:.3f}, Verb Prec@5 {:.3f}\n'.format(verb_top1.avg, verb_top5.avg)
            msg += '  Noun Prec@1 {:.3f}, Noun Prec@5 {:.3f}'.format(noun_top1.avg, noun_top5.avg)
            logger.info(msg)

    # Print out message
    all_skip = np.concatenate(all_skip, axis=0)
    all_ssim = np.concatenate(all_ssim, axis=0)
    all_time = np.concatenate(all_time, axis=0)
    msg = 'Overall results:\n'
    msg += '  Prec@1 {:.3f}, Prec@5 {:.3f}\n'.format(top1.avg, top5.avg)
    msg += '  Verb Prec@1 {:.3f}, Verb Prec@5 {:.3f}\n'.format(verb_top1.avg, verb_top5.avg)
    msg += '  Noun Prec@1 {:.3f}, Noun Prec@5 {:.3f}\n'.format(noun_top1.avg, noun_top5.avg)
    msg += '  Total frames {}, Skipped frames {}'.format(all_skip.size, all_skip.sum())
    logger.info(msg)

    # Collect metrics
    test_metrics = {'top1': top1.avg,
                    'top5': top5.avg,
                    'verb_top1': verb_top1.avg,
                    'verb_top5': verb_top5.avg,
                    'noun_top1': noun_top1.avg,
                    'noun_top5': noun_top1.avg,
                    }
    results = {
        'test_metrics': test_metrics,
        'all_output': all_output,
        'all_skip': all_skip,
        'all_ssim': all_ssim,
        'all_time': all_time,
    }
    return results


def test_without_gt(model, device, test_loader):
    """Test on the test set without groundtruth labels
    """
    model_name = type(model).__name__
    assert model_name in ['Pipeline6', 'Pipeline8']
    assert test_loader.dataset.name == 'epic_kitchens', \
        'Unsupported dataset: {}'.format(test_loader.dataset.dataset_name)

    # Prepare for json output
    uid_lst = test_loader.dataset.list_file.index.values
    results = {
        "version": "0.1",
        "challenge": "action_recognition",
        "results": {},
    }

    # Test
    all_skip, all_time, all_ssim = [], [], []
    for i, (sample, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Inference
        sample = {k: v.to(device) for k, v in sample.items()}
        if model_name == 'Pipeline6':
            output, extra_output = model(sample)
        elif model_name == 'Pipeline8':
            output, _, _, extra_output = model(sample, get_extra=True)
        assert output[0].shape[0] == 1, 'Only support batch_size=1'

        verb_output = output[0][0].cpu().numpy()
        noun_output = output[1][0].cpu().numpy()

        # Collect prediction
        uid = str(uid_lst[i])
        results["results"][uid] = {
            'verb': {str(k): float(verb_output[k]) for k in range(len(verb_output))},
            'noun': {str(k): float(noun_output[k]) for k in range(len(noun_output))},
        }

        # Collect extra results
        all_skip.append(extra_output['skip'])
        all_time.append(extra_output['time'])
        if isinstance(extra_output['ssim'], np.ndarray):
            all_ssim.append(extra_output['ssim'])
        else:
            all_ssim.append(extra_output['ssim'].cpu().numpy())

    # Print out message
    msg = '  Total frames {}, Skipped frames {}'.format(len(all_skip), sum(all_skip))
    logger.info(msg)

    all_skip = np.concatenate(all_skip, axis=0)
    all_ssim = np.concatenate(all_ssim, axis=0)
    all_time = np.concatenate(all_time, axis=0)
    extra_results = {
        'all_skip': all_skip,
        'all_ssim': all_ssim,
        'all_time': all_time,
    }

    return results, extra_results
