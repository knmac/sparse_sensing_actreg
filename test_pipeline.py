"""Run the full pipeline with both time and space sampler
"""
import torch
from tqdm import tqdm

import src.utils.logging as logging
from src.utils.metrics import AverageMeter, accuracy, multitask_accuracy

logger = logging.get_logger(__name__)


def test(model, device, test_loader, args, test_mode='by_segment'):
    assert test_loader.dataset.name == 'epic_kitchens', \
        'Unsupported dataset: {}'.format(test_loader.dataset.dataset_name)
    assert type(model).__name__ == 'Pipeline6'

    # Switch model to eval mode
    model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)
    model.eval()

    # Test
    with torch.no_grad():
        if test_mode == 'by_segment':
            _test_by_segment(model, device, test_loader)
        elif test_mode == 'by_vid':
            _test_by_vid()
        else:
            raise NotImplementedError


def _test_by_segment(model, device, test_loader):
    """Test each segment as a whole (each video contains multiple segments)
    Do not call directly. Use test() instead
    """
    # Prepare metrics
    top1 = AverageMeter()
    top5 = AverageMeter()
    verb_top1 = AverageMeter()
    verb_top5 = AverageMeter()
    noun_top1 = AverageMeter()
    noun_top5 = AverageMeter()

    # Test
    for i, (sample, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Forward
        sample = {k: v.to(device) for k, v in sample.items()}
        target = {k: v.to(device) for k, v in target.items()}
        output, extra_output = model(sample)

        # Compute metrics
        batch_size = sample[model.module.modality[0]].size(0)
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

    # Print out message
    msg = 'Testing results:\n'
    msg += '  Prec@1 {:.3f}, Prec@5 {:.3f}\n'.format(top1.avg, top5.avg)
    msg += '  Verb Prec@1 {:.3f}, Verb Prec@5 {:.3f}\n'.format(verb_top1.avg, verb_top5.avg)
    msg += '  Noun Prec@1 {:.3f}, Noun Prec@5 {:.3f}'.format(noun_top1.avg, noun_top5.avg)
    logger.info(msg)

    # Collect metrics
    test_metrics = {'test_acc': top1.avg,
                    'test_verb_acc': verb_top1.avg,
                    'test_noun_acc': noun_top1.avg}
    return test_metrics


def _test_by_vid():
    raise NotImplementedError
