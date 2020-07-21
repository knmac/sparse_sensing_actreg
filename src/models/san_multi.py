"""Wrapper for multiple modalities with SAN backbone. Only for feature extraction
"""
import sys
import os
from collections import OrderedDict

import torch
from torch import nn

from .san import SAN, Bottleneck
from .base_model import BaseModel

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class SANMulti(BaseModel):

    def __init__(self, device, num_segments, modality,
                 san_sa_type, san_layers, san_kernels,
                 san_pretrained_weights=None, new_length=None, **kwargs):
        super(SANMulti, self).__init__(device)
        # self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                self.new_length[m] = 1 if (m == "RGB" or m == "Spec") else 5
        else:
            self.new_length = new_length

        # parameters of SAN backbone
        self.san_sa_type = san_sa_type
        self.san_layers = san_layers
        self.san_kernels = san_kernels

        # Get the pretrained weight and convert to dictionary if neccessary
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # if (san_pretrained_weights is not None) and (not os.path.isfile(san_pretrained_weights)):
        #     san_pretrained_weights = os.path.join(root, san_pretrained_weights)
        # self.san_pretrained_weights = san_pretrained_weights
        if san_pretrained_weights is not None:
            # If str -> all modalities are init using the same weight
            if isinstance(san_pretrained_weights, str):
                if not os.path.isfile(san_pretrained_weights):
                    san_pretrained_weights = os.path.join(root, san_pretrained_weights)
                self.san_pretrained_weights = {m: san_pretrained_weights for m in modality}
            # If dict -> each modality is init differently
            elif isinstance(san_pretrained_weights, dict):
                self.san_pretrained_weights = {}
                for m in modality:
                    if not os.path.isfile(san_pretrained_weights[m]):
                        san_pretrained_weights[m] = os.path.join(root, san_pretrained_weights[m])
                self.san_pretrained_weights = san_pretrained_weights
            else:
                raise 'san_pretrained_weights must be string or dictionary'
        else:
            self.san_pretrained_weights = {m: None for m in modality}

        # Prepare SAN basemodels
        self._load_weight_later = []
        self._prepare_base_model()

        # Prepare the flow and spec modalities by replacing the 1st layer
        is_flow = any(m == 'Flow' for m in self.modality)
        is_spec = any(m == 'Spec' for m in self.modality)
        if is_flow:
            logger.info('Converting the ImageNet model to a flow init model')
            self.base_model['Flow'] = self._construct_flow_model(self.base_model['Flow'])
            logger.info('Done. Flow model ready...')
        if is_spec:
            logger.info('Converting the ImageNet model to a spectrogram init model')
            self.base_model['Spec'] = self._construct_spec_model(self.base_model['Spec'])
            logger.info('Done. Spec model ready.')

        # Load the weights if could not load before
        if len(self._load_weight_later) != 0:
            for m in self._load_weight_later:
                if os.path.isfile(self.san_pretrained_weights[m]):
                    checkpoint = torch.load(self.san_pretrained_weights[m])

                    # Remove `module.` from keys
                    state_dict = {k.replace('module.', ''): v
                                  for k, v in checkpoint['state_dict'].items()}
                    self.base_model[m].load_state_dict(state_dict, strict=True)
                    logger.info('Reloaded pretrained weight for SAN modality: {}'.format(m))
                else:
                    logger.info('Not loading pretrained model for modality {}!'.format(m))
        del self._load_weight_later

        # Remove the last fc layer
        for m in self.modality:
            last_layer_name = 'fc'
            delattr(self.base_model[m], last_layer_name)

        # Add base models as modules
        for m in self.modality:
            self.add_module(m.lower(), self.base_model[m])

    def _prepare_base_model(self):
        """Prepare SAN basemodel for each of the modality"""
        self.base_model = OrderedDict()
        self.input_size = OrderedDict()
        self.input_mean = OrderedDict()
        self.input_std = OrderedDict()
        self.feature_dim = 2048  # Feature dimension before final fc layer

        for m in self.modality:
            # Build SAN models
            self.base_model[m] = SAN(
                sa_type=self.san_sa_type,
                block=Bottleneck,
                layers=self.san_layers,
                kernels=self.san_kernels,
                num_classes=1000,  # Final fc will be removed later
            )
            self.input_size[m] = 224
            self.input_std[m] = [1]

            if m == 'Flow':
                self.input_mean[m] = [128]
            elif m == 'RGBDiff':
                self.input_mean[m] = self.input_mean[m] * (1 + self.new_length[m])
            elif m == 'RGB':
                self.input_mean[m] = [104, 117, 128]

            # Load pretrained weights
            if os.path.isfile(self.san_pretrained_weights[m]):
                logger.info('Loading pretrained weight for SAN modality: {}'.format(m))
                checkpoint = torch.load(self.san_pretrained_weights[m])

                # Remove `module.` from keys
                state_dict = {k.replace('module.', ''): v
                              for k, v in checkpoint['state_dict'].items()}
                try:
                    self.base_model[m].load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    logger.info('Cannot load. Will conver and load later...')
                    self._load_weight_later.append(m)
            else:
                logger.info('Not loading pretrained model for modality {}!'.format(m))

    def _construct_flow_model(self, base_model):
        """Covert ImageNet model to flow init model"""
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['Flow'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length['Flow'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length['Flow'], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_spec_model(self, base_model):
        """Convert ImageNet model to spectrogram init model"""
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['Spec'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).contiguous()

        new_conv = nn.Conv2d(self.new_length['Spec'], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)

        # replace the avg pooling at the end, so that it matches the spectrogram dimensionality (256x256)
        # NOTE: no need because using AdaptiveAvgPool2d
        # pool_layer = getattr(self.base_model['Spec'], 'global_pool')
        # new_avg_pooling = nn.AvgPool2d(8, stride=pool_layer.stride, padding=pool_layer.padding)
        # setattr(self.base_model['Spec'], 'global_pool', new_avg_pooling)

        return base_model

    def forward(self, x):
        """Forward to get the feature instead of getting the classification

        Args:
            x: dictionary inputs of multiple modalities

        Return:
            out_feat: concatenated feature output
        """
        concatenated = []

        # Get the output for each modality
        for m in self.modality:
            if (m == 'RGB'):
                channel = 3
            elif (m == 'Flow'):
                channel = 2
            elif (m == 'Spec'):
                channel = 1
            sample_len = channel * self.new_length[m]

            base_model = getattr(self, m.lower())
            base_out = base_model(x[m].view((-1, sample_len) + x[m].size()[-2:]))

            base_out = base_out.view(base_out.size(0), -1)
            concatenated.append(base_out)
        out_feat = torch.cat(concatenated, dim=1)
        return out_feat

    def freeze_fn(self, freeze_mode):
        """Copied from tbn model"""
        if freeze_mode == 'modalities':
            for m in self.modality:
                logger.info('Freezing ' + m + ' stream\'s parameters')
                base_model = getattr(self, m.lower())
                for param in base_model.parameters():
                    param.requires_grad_(False)

        elif freeze_mode == 'partialbn_parameters':
            for mod in self.modality:
                count = 0
                logger.info("Freezing BatchNorm2D parameters except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown parameters update in frozen mode
                            m.weight.requires_grad_(False)
                            m.bias.requires_grad_(False)

        elif freeze_mode == 'partialbn_statistics':
            for mod in self.modality:
                count = 0
                logger.info("Freezing BatchNorm2D statistics except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown running statistics update in frozen mode
                            m.eval()
        elif freeze_mode == 'bn_statistics':
            for mod in self.modality:
                logger.info("Freezing BatchNorm2D statistics.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # shutdown running statistics update in frozen mode
                        m.eval()
        else:
            raise ValueError('Unknown mode for freezing the model: {}'.format(freeze_mode))

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = {k: v * 256 // 224 for k, v in self.input_size.items()}
        return scale_size

    def get_param_groups(self):
        """Wrapper to get param_groups for optimizer
        """
        if len(self.modality) > 1:
            param_groups = []
            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.rgb.parameters())})
            except AttributeError:
                pass

            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.flow.parameters()), 'lr': 0.001})
            except AttributeError:
                pass

            try:
                param_groups.append({'params': filter(lambda p: p.requires_grad, self.spec.parameters())})
            except AttributeError:
                pass

            param_groups.append({'params': filter(lambda p: p.requires_grad, self.fusion_classification_net.parameters())})
        else:
            param_groups = filter(lambda p: p.requires_grad, self.parameters())
        return param_groups
