import os
import time

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from src.utils.misc import MiscUtils
import src.utils.logging as logging
from src.utils.metrics import AverageMeter, accuracy, multitask_accuracy

logger = logging.get_logger(__name__)


def train_val(model, device, criterion, train_loader, val_loader, train_params, args):
    """Training and validation routine. It will call val() automatically
    """
    # Get param_groups for optimizer
    param_groups = model.get_param_groups()

    # Create optimizer
    if train_params['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, **train_params['optim_params'])
    elif train_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups, **train_params['optim_params'])
    else:
        raise NotImplementedError

    # Create scheduler
    scheduler = MultiStepLR(optimizer, train_params['lr_steps'], gamma=0.1)

    # Summary writer
    sum_writer = SummaryWriter(log_dir=args.logdir)

    # TODO: stats_dict ?

    # Setup training starting point
    start_epoch, lr, model, best_prec1 = _setup_training(model, optimizer, train_params, args)

    # Train with multiple GPUs
    model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)

    # -------------------------------------------------------------------------
    # Go through all epochs
    for epoch in range(start_epoch, train_params['n_epochs']):
        scheduler.step()

        # Training phase
        run_iter = epoch * len(train_loader)
        _train_one_epoch(
            model, device, criterion, train_loader, optimizer, sum_writer,
            epoch, run_iter, train_params)

        # Validation phase
        # TODO: check the case when there's no validation set
        if ((epoch + 1) % train_params['eval_freq'] == 0) or ((epoch + 1) == train_params['n_epochs']):
            val_metrics = validate(model, device, criterion, val_loader,
                                   sum_writer, run_iter+len(train_loader))

            # Remember best prec@1 and save checkpoint
            prec1 = val_metrics['val_acc']
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            MiscUtils.save_progress(model, optimizer, args.logdir, best_prec1, epoch, is_best)

    # Done training
    sum_writer.close()


def _setup_training(model, optimizer, train_params, args):
    """Set up training model, optimizer and get starting epoch and lr

    Args:
        model: model for resuming or loading pretrained
        optimizer: optimizer for resuming
        train_params: training parameters
        args: extra runtime parameters

    Return:
        start_epoch: epoch to start the training at
        lr: learning rate to start at
        model: loaded model by `from_pretrained` or `resume`
        best_prec1: current best prec@1
    """
    train_mode = args.train_mode
    logdir = args.logdir
    # pretrained_model_path = args.pretrained_model_path

    # By default, start_epoch = 0, lr from train parameters
    start_epoch = 0
    best_prec1 = 0
    lr = train_params['optim_params']['lr']

    # Setup training starting point
    if train_mode == 'from_scratch':
        logger.info('Start training from scratch')
    elif train_mode == 'from_pretrained':
        # logger.info('Load pretrained weights')
        # model.load_model(pretrained_model_path)
        # TODO: only support pretrained flow weight for now
        logger.info('Initialize Flow stream from Kinetics')
        pretrained = 'pretrained/kinetics_tsn_flow.pth.tar'
        if not os.path.isfile(pretrained):
            root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
            pretrained = os.path.join(root, pretrained)
        state_dict = torch.load(pretrained)
        for k, v in state_dict.items():
            state_dict[k] = torch.squeeze(v, dim=0)
        try:
            base_model = getattr(model, 'flow')
        except AttributeError:
            base_model = getattr(model.light_model, 'flow')
        base_model.load_state_dict(state_dict, strict=False)
    elif train_mode == 'resume':
        logger.info('Resume training from a checkpoint')
        prefix = MiscUtils.get_lastest_checkpoint(logdir)
        lr, start_epoch, best_prec1 = MiscUtils.load_progress(model, optimizer, prefix)
    else:
        raise ValueError('Unsupported train_mode: {}'.format(train_mode))

    # Freeze stream weights (leaves only fusion and classification trainable)
    if train_params['freeze']:
        model.freeze_fn('modalities')

    # Freeze batch normalisation layers except the first
    if train_params['partialbn']:
        model.freeze_fn('partialbn_parameters')
    return start_epoch, lr, model, best_prec1


def _train_one_epoch(model, device, criterion, train_loader, optimizer,
                     sum_writer, epoch, run_iter, train_params):
    """Train one single epoch
    """
    dataset = train_loader.dataset.name
    clip_gradient = float(train_params['clip_gradient'])

    # Switch to train mode
    model.train()

    if train_params['partialbn']:
        model.module.freeze_fn('partialbn_statistics')
    if train_params['freeze']:
        model.module.freeze_fn('bn_statistics')

    # Prepare metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if dataset == 'epic_kitchens':
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        noun_top1 = AverageMeter()
        noun_top5 = AverageMeter()

    # Training loop
    end = time.time()
    for i, (sample, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Place input sample on the correct device for all modalities
        for k in sample.keys():
            sample[k] = sample[k].to(device)

        # Forward
        output = model(sample)

        # Compute metrics
        batch_size = sample[model.module.modality[0]].size(0)
        if dataset != 'epic_kitchens':
            target = target.to(device)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
        else:
            target = {k: v.to(device) for k, v in target.items()}
            loss_verb = criterion(output[0], target['verb'])
            loss_noun = criterion(output[1], target['noun'])
            loss = 0.5 * (loss_verb + loss_noun)
            verb_losses.update(loss_verb.item(), batch_size)
            noun_losses.update(loss_noun.item(), batch_size)

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
        losses.update(loss.item(), batch_size)
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), clip_gradient)
            if total_norm > clip_gradient:
                logger.info('Clipping gradient: %f with coef %f' %
                            (total_norm, clip_gradient / total_norm))
        optimizer.step()
        run_iter += 1

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print out message
        if i % train_params['print_freq'] == 0:
            _lr = optimizer.param_groups[-1]['lr']

            sum_writer.add_scalar('data/epochs', epoch, run_iter)
            sum_writer.add_scalar('data/learning_rate', _lr, run_iter)

            msg_prefix = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            msg_prefix += 'lr={:.5f}, batch_time={:.3f}, data_time={:.3f}\n'.format(
                _lr, batch_time.avg, data_time.avg)

            log_content = {'losses': losses, 'top1': top1, 'top5': top5}
            if dataset == 'epic_kitchens':
                log_content.update({
                    'verb_losses': verb_losses, 'verb_top1': verb_top1, 'verb_top5': verb_top5,
                    'noun_losses': noun_losses, 'noun_top1': noun_top1, 'noun_top5': noun_top5,
                })

            _log_message('training', sum_writer, run_iter, log_content, msg_prefix)

    # Collect training metrics
    if dataset != 'epic_kitchens':
        training_metrics = {'train_loss': losses.avg, 'train_acc': top1.avg}
    else:
        training_metrics = {'train_loss': losses.avg,
                            'train_noun_loss': noun_losses.avg,
                            'train_verb_loss': verb_losses.avg,
                            'train_acc': top1.avg,
                            'train_verb_acc': verb_top1.avg,
                            'train_noun_acc': noun_top1.avg}
    return training_metrics


def validate(model, device, criterion, val_loader, sum_writer=None, run_iter=None):
    """Validate a trained model
    """
    dataset = val_loader.dataset.name

    # Swith to eval mode
    model.eval()

    with torch.no_grad():
        # Prepare metrics
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if dataset == 'epic_kitchens':
            verb_losses = AverageMeter()
            noun_losses = AverageMeter()
            verb_top1 = AverageMeter()
            verb_top5 = AverageMeter()
            noun_top1 = AverageMeter()
            noun_top5 = AverageMeter()

        # Validation loop
        end = time.time()
        for i, (sample, target) in enumerate(val_loader):
            # Place input sample on the correct device for all modalities
            for k in sample.keys():
                sample[k] = sample[k].to(device)

            # Forward
            output = model(sample)

            # Compute metrics
            batch_size = sample[model.module.modality[0]].size(0)
            if dataset != 'epic_kitchens':
                target = target.to(device)
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
            else:
                target = {k: v.to(device) for k, v in target.items()}
                loss_verb = criterion(output[0], target['verb'])
                loss_noun = criterion(output[1], target['noun'])
                loss = 0.5 * (loss_verb + loss_noun)
                verb_losses.update(loss_verb.item(), batch_size)
                noun_losses.update(loss_noun.item(), batch_size)

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

            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # Print out message
        msg_prefix = 'Testing results:\n'
        log_content = {'losses': losses, 'top1': top1, 'top5': top5}

        if dataset == 'epic_kitchens':
            log_content.update({
                'verb_losses': verb_losses, 'verb_top1': verb_top1, 'verb_top5': verb_top5,
                'noun_losses': noun_losses, 'noun_top1': noun_top1, 'noun_top5': noun_top5,
            })
        _log_message('validation', sum_writer, run_iter, log_content, msg_prefix)

        # Collect validation metrics
        if dataset != 'epic_kitchens':
            val_metrics = {'val_loss': losses.avg, 'val_acc': top1.avg}
        else:
            val_metrics = {'val_loss': losses.avg,
                           'val_noun_loss': noun_losses.avg,
                           'val_verb_loss': verb_losses.avg,
                           'val_acc': top1.avg,
                           'val_verb_acc': verb_top1.avg,
                           'val_noun_acc': noun_top1.avg}
        return val_metrics


def _log_message(phase, sum_writer, run_iter, data, msg_prefix=''):
    """Wrapper to print message and writer summary"""
    msg = msg_prefix

    if (sum_writer is not None) and (run_iter is not None):
        sum_writer.add_scalars('data/loss', {phase: data['losses'].avg}, run_iter)
        sum_writer.add_scalars('data/prec/top1', {phase: data['top1'].avg}, run_iter)
        sum_writer.add_scalars('data/prec/top5', {phase: data['top5'].avg}, run_iter)

    msg += '  Loss {:.4f}, Prec@1 {:.3f}, Prec@5 {:.3f}\n'.format(
        data['losses'].avg, data['top1'].avg, data['top5'].avg)

    try:
        if (sum_writer is not None) and (run_iter is not None):
            sum_writer.add_scalars('data/verb/loss', {phase: data['verb_losses'].avg}, run_iter)
            sum_writer.add_scalars('data/noun/loss', {phase: data['noun_losses'].avg}, run_iter)
            sum_writer.add_scalars('data/verb/prec/top1', {phase: data['verb_top1'].avg}, run_iter)
            sum_writer.add_scalars('data/verb/prec/top5', {phase: data['verb_top5'].avg}, run_iter)
            sum_writer.add_scalars('data/noun/prec/top1', {phase: data['noun_top1'].avg}, run_iter)
            sum_writer.add_scalars('data/noun/prec/top5', {phase: data['noun_top5'].avg}, run_iter)

        msg += '  Verb Loss {:.4f}, Verb Prec@1 {:.3f}, Verb Prec@5 {:.3f}\n'.format(
            data['verb_losses'].avg, data['verb_top1'].avg, data['verb_top5'].avg)
        msg += '  Noun Loss {:.4f}, Noun Prec@1 {:.3f}, Noun Prec@5 {:.3f}'.format(
            data['noun_losses'].avg, data['noun_top1'].avg, data['noun_top5'].avg)
    except KeyError:
        pass

    logger.info(msg)
