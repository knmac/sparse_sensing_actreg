"""Simple simulation for time sampling"""
import sys
import os

# import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from skimage.transform import resize

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.factories import ModelFactory
from src.factories import DatasetFactory
from src.utils.load_cfg import ConfigLoader
from src.utils.misc import MiscUtils
# from src.utils.metrics import accuracy, multitask_accuracy
from src.models.pytorch_ssim.ssim import SSIM

# from tools.complexity import (get_model_complexity_info,
#                               is_supported_instance,
#                               flops_to_string,
#                               get_model_parameters_number)

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("whitegrid", {'axes.grid': False})


def actreg_nosample(model, sample):
    # Feed to the feature extraction
    x = model.light_model(sample)

    # Feed to the actreg model with appropriate number of frames
    model.actreg_model.rnn.flatten_parameters()
    x = model.actreg_model.relu(model.actreg_model.fc1(x))
    x = x.view(-1, model.num_segments, model.actreg_model.rnn_input_size)
    x, _ = model.actreg_model.rnn(x, None)
    x = model.actreg_model.relu(x)

    output_list_nosampled = []
    for t in range(model.num_segments):
        output_list_nosampled.append(model.actreg_model.classify(x[:, t, :]))

    score_verb_nosampled = np.stack([F.softmax(item[0], dim=1)[0].cpu()
                                     for item in output_list_nosampled], axis=0)
    score_noun_nosampled = np.stack([F.softmax(item[1], dim=1)[0].cpu()
                                     for item in output_list_nosampled], axis=0)
    return score_verb_nosampled, score_noun_nosampled


def actreg_withsample(model, sample, ssim_list, theta):
    # Feed to the feature extraction
    x = model.light_model(sample)

    # Sample to ignore frames with low SSIM
    x = x[ssim_list > theta]
    num_sampled = x.shape[0]

    # Feed to the actreg model with appropriate number of frames
    model.actreg_model.rnn.flatten_parameters()
    x = model.actreg_model.relu(model.actreg_model.fc1(x))
    x = x.view(-1, num_sampled, model.actreg_model.rnn_input_size)
    x, _ = model.actreg_model.rnn(x, None)
    x = model.actreg_model.relu(x)

    output_list_sampled = []
    t1, t2 = 0, 0
    while t1 < model.num_segments:
        if ssim_list[t1] > theta:  # not removed frame -> compute new result
            output_list_sampled.append(model.actreg_model.classify(x[:, t2, :]))
            t2 += 1
        else:  # removed frame -> reuse prev result
            output_list_sampled.append(output_list_sampled[-1])
        t1 += 1
    score_verb_sampled = np.stack([F.softmax(item[0], dim=1)[0].cpu()
                                   for item in output_list_sampled], axis=0)
    score_noun_sampled = np.stack([F.softmax(item[1], dim=1)[0].cpu()
                                   for item in output_list_sampled], axis=0)
    return score_verb_sampled, score_noun_sampled


def main():
    dataset_cfg = 'configs/dataset_cfgs/epickitchens_short.yaml'
    train_cfg = 'configs/train_cfgs/train_san_freeze_short.yaml'
    model_cfg = 'configs/model_cfgs/pipeline2_rgbspec_san19pairfreeze_actreggru_halluconvlstm.yaml'

    weight = 'saved_models/san19freeze_halluconvlstm_actreggru/dim32_layer1_nsegment3/epoch_00049.model'
    model_cfg_mod = None
    # weight = 'saved_models/san19freeze_halluconvlstm_actreggru/dim32_layer1_nsegment10/epoch_00049.model'
    # model_cfg_mod = {'num_segments': 10, 'hallu_model_cfg': 'exp_cfgs/haluconvlstm_32_1.yaml'}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and data loader
    # Load configurations
    model_name, model_params = ConfigLoader.load_model_cfg(model_cfg)
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(dataset_cfg)
    train_params = ConfigLoader.load_train_cfg(train_cfg)

    if model_cfg_mod is not None:
        model_params.update(model_cfg_mod)

    dataset_params.update({
        'modality': model_params['modality'],
        'num_segments': model_params['num_segments'],
        'new_length': model_params['new_length'],
    })

    # Build model
    model_factory = ModelFactory()
    model = model_factory.generate(model_name, device=device,
                                   model_factory=model_factory, **model_params)
    model.load_model(weight)
    model = model.to(device)
    model.eval()

    # Get training augmentation and transforms
    train_augmentation = MiscUtils.get_train_augmentation(model.modality, model.crop_size)
    train_transform, val_transform = MiscUtils.get_train_val_transforms(
        modality=model.modality,
        input_mean=model.input_mean,
        input_std=model.input_std,
        scale_size=model.scale_size,
        crop_size=model.crop_size,
        train_augmentation=train_augmentation,
    )

    # Data loader
    dataset_factory = DatasetFactory()
    loader_params = {
        'batch_size': train_params['batch_size'],
        'num_workers': train_params['num_workers'],
        'pin_memory': True,
    }

    val_dataset = dataset_factory.generate(dataset_name, mode='val',
                                           transform=val_transform,
                                           **dataset_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)

    # Go through the dataset
    ssim_criterion = SSIM(window_size=3, channel=32)
    correct_verb_nosampled = []
    correct_noun_nosampled = []
    correct_verb_sampled = []
    correct_noun_sampled = []
    with torch.no_grad():
        for i, (sample, target) in enumerate(val_loader):
            sample = {k: v.to(device) for k, v in sample.items()}
            target = {k: v.to(device) for k, v in target.items()}
            target_verb, target_noun = target['verb'].item(), target['noun'].item()

            # Get attention and hallucination
            model(sample)
            attn = model._attn[0]
            hallu = model._hallu[0]

            # Compute ssim
            ssim_list = np.zeros(model.num_segments)
            for t in range(1, model.num_segments):
                ssim = -ssim_criterion(attn[t].unsqueeze(dim=0),
                                       hallu[t-1].unsqueeze(dim=0)).item()
                ssim_list[t] = ssim

            # Compute score for each frame
            score_verb_nosampled, score_noun_nosampled = actreg_nosample(model, sample)
            score_verb_sampled, score_noun_sampled = actreg_withsample(model, sample, ssim_list, -0.25)

            # Collect results
            correct_verb_nosampled.append(score_verb_nosampled.argmax(axis=1) == target_verb)
            correct_noun_nosampled.append(score_noun_nosampled.argmax(axis=1) == target_noun)
            correct_verb_sampled.append(score_verb_sampled.argmax(axis=1) == target_verb)
            correct_noun_sampled.append(score_noun_sampled.argmax(axis=1) == target_noun)

            if i % 100 == 0:
                print('{}/{}'.format(i, len(val_loader)))

    print('Accuracy without sampling')
    print('- Verb: {:.4f}%'.format(np.concatenate(correct_verb_nosampled).mean() * 100))
    print('- Noun: {:.4f}%'.format(np.concatenate(correct_noun_nosampled).mean() * 100))
    print('Accuracy with sampling')
    print('- Verb: {:.4f}%'.format(np.concatenate(correct_verb_sampled).mean() * 100))
    print('- Noun: {:.4f}%'.format(np.concatenate(correct_noun_sampled).mean() * 100))


if __name__ == '__main__':
    main()
