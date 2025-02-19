# Source code modified from: https://github.com/HRNet/HRNet-Image-Classification/blob/master/lib/models/cls_hrnet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools
from typing import Dict, List, Optional, Union, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models import ResNet152_Weights, resnet152
from .utils import load_pth

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], 
                                       momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg: dict, **kwargs):
        super(HighResolutionNet, self).__init__()
        
        self.output_each_stage = cfg.get('OUTPUT_EACH_STAGE', False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # Classification Head
        # self.incre_modules, self.downsamp_modules, \
        #     self.final_layer = self._make_head(pre_stage_channels)

        # self.classifier = nn.Linear(2048, 1000)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        
        if self.output_each_stage:
            out_list = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        if self.output_each_stage:
            out_list.append(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        if self.output_each_stage:
            out_list.extend(y_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        if self.output_each_stage:
            out_list.extend(y_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        out_shape = y_list[0].shape[-2:]
        
        if self.output_each_stage:
            out_list.extend(y_list)
        else:
            out_list = y_list
        
        out_list_interpolated = []
        # return y_list
        for y in out_list:
            out_list_interpolated.append(
                F.interpolate(
                    y, 
                    size=out_shape, 
                    mode='bilinear', 
                    align_corners=True
                )
            )
        
        return torch.cat(out_list_interpolated, 1)

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = load_pth(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model



class DecoderModule(nn.Module):
    def __init__(self, num_channels: int=720, num_blocks: int=2):
        super(DecoderModule, self).__init__()

        self.num_blocks = num_blocks
        self.module_list = nn.ModuleList()
        for _ in range(num_blocks):
            self.module_list.append(nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x_up.clone()
        
        for module in self.module_list:
            x = module(x)
        
        return x + x_up



class ImageDecoderHead(nn.Module):
    
    def __init__(self, in_channels: int=720, num_classes: int=3, num_blocks: int=2):
        super(ImageDecoderHead, self).__init__()

        self.in_channels = in_channels
        
        self.layer1 = DecoderModule(num_channels=in_channels, num_blocks=num_blocks)
        self.layer2 = DecoderModule(num_channels=in_channels, num_blocks=num_blocks)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # for residual connection between encoder output and final output
        x_upsampled = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x += x_upsampled # residual connection from encoder output
        x = self.classifier(x)
        return x
    
    def reinit_classifier(self, num_classes: int=3):
        self.classifier = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        return self        


# ProjectionHead implementation inspired by official TF code https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/tf2/model.py#L157
# per paper, only use one hidden layer and do not apply a non-linearity to the output embeddings
# z_i = W^{(2)} \sigma(W^{(1)} h_i)
class ProjectionHead(nn.Module):
    
    def __init__(self, in_channels: int=720, num_hiddens: int=1, embedding_dim: int=128):
        super(ProjectionHead, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.hiddens = nn.ModuleList([])
        for _ in range(num_hiddens):
            self.hiddens.append(nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True)
            ))
        
        self.output = nn.Sequential(
            nn.Linear(in_channels, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.gap(x).view(x.size(0), -1) # reshape to (batch_size, num_channels)
        
        for hidden_layer in self.hiddens:
            x = hidden_layer(x) + x

        return self.output(x)



class SimpleImageDecoderHead(nn.Module):
    '''
    Image decoder that simply upsamples the input tensor by a factor of 4 and 
    applies a 1x1 convolution to reduce the number of channels to the number of 
    classes.
    '''
    
    def __init__(self, in_channels: int=720, num_classes: int=3, output_size: Tuple[int, int]=(256, 256)):
        super(SimpleImageDecoderHead, self).__init__()

        self.in_channels = in_channels
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.output_size = output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return self.classifier(x)
    
    def reinit_classifier(self, num_classes: int=3):
        self.classifier = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        return self



class SEBlock(nn.Module):
    
    def __init__(self, channels: int, reduction_factor: int=16):
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_factor, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEImageDecoderHead(nn.Module):
    
    def __init__(self, in_channels: int=720, num_classes: int=3, hidden_dim: int=192, num_hiddens: int=4, num_bottleneck_blocks: int=1):
        super(SEImageDecoderHead, self).__init__()
        
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.num_hidden = num_hiddens

        self.seblock = SEBlock(in_channels)
        self.bottleneck = nn.ModuleList([])
        for i in range(num_bottleneck_blocks):
            in_dim = in_channels if i == 0 else hidden_dim
            self.bottleneck.append(nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                SEBlock(hidden_dim),
            ))
        
        self.conv1x1s = nn.ModuleList([])
        self.conv3x3s = nn.ModuleList([])
        self.conv5x5s = nn.ModuleList([])
        self.conv7x7s = nn.ModuleList([])
        self.seblocks = nn.ModuleList([])
        for _ in range(num_hiddens):
            self.conv1x1s.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ))
            self.conv3x3s.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ))
            self.conv5x5s.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ))
            self.conv7x7s.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ))
            self.seblocks.append(SEBlock(hidden_dim))
        
        self.classifer = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.seblock(x)
        for i, bottleneck in enumerate(self.bottleneck):
            if i == 0:
                x = bottleneck(x)
            else:
                x = bottleneck(x) + x
                
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x_int = x.clone()
        for i in range(self.num_hidden):
            x1 = self.conv1x1s[i](x)
            x3 = self.conv3x3s[i](x)
            x5 = self.conv5x5s[i](x)
            x7 = self.conv7x7s[i](x)
            x_convd = x1 + x3 + x5 + x7 + x
            if i != 0:
                x_convd += x_int
            x = self.seblocks[i](x_convd)
        
        return self.classifer(x)


class HRNetSegmentationModel(nn.Module):
    
    def __init__(self, 
        config: dict, 
        img_decoder_head: bool=True,
        use_simple_decoder: bool=True, # if True, use SimpleImageDecoderHead, else use ImageDecoderHead with multiple blocks
        use_se_decoder: bool=False,
        img_decoder_activation: str='sigmoid',
        num_classes: int=3, 
        aux_simclr_head: bool=False,
        unet_like_decoder: bool=False,
    ):
        
        if img_decoder_activation not in ['sigmoid', 'softmax', 'none', None]:
            raise ValueError('Invalid value for `img_decoder_activation`. Must be one of ["sigmoid", "softmax", "none", None].')

        super(HRNetSegmentationModel, self).__init__()
        
        if unet_like_decoder:
            self.encoder_output_channels = sum([sum(config['STAGE{}'.format(i)]['NUM_CHANNELS']) for i in range(2, 5)]) + 256 # + 256 to account for stem in HRNET_W18, may be different for HRNET_248
            config['OUTPUT_EACH_STAGE'] = True
        else:
            self.encoder_output_channels = sum(config['STAGE4']['NUM_CHANNELS'])
            config['OUTPUT_EACH_STAGE'] = False
        
        self.num_classes = num_classes
        self.config = config
        
        self.encoder = get_cls_net(config)
        
        
        
        # if not (img_decoder_head or aux_simclr_head):
            # raise ValueError('At least one of `img_decoder_head` or `aux_simclr_head` must be True.')
        
        self.decoder = None
        if img_decoder_head:
            if use_simple_decoder:
                self.decoder = SimpleImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes)
            elif use_se_decoder:
                self.decoder = SEImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes)
            else:
                self.decoder = ImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes, num_blocks=config['IMAGE_DECODER']['NUM_BLOCKS'])
            
            # self.img_decoder_activation = nn.Identity()
            if img_decoder_activation == 'sigmoid':
                self.img_decoder_activation = nn.Sigmoid()
            elif img_decoder_activation == 'softmax':
                self.img_decoder_activation = nn.Softmax(dim=1)
            else:
                self.img_decoder_activation = nn.Identity()
                    
        self.projection_head = None
        if aux_simclr_head:
            self.projection_head = ProjectionHead(in_channels=self.encoder_output_channels)
    
    
    
    def load_encoder_weights(self, state_dict: dict):
        for key in list(state_dict.keys()):
            if key.split('.')[0] == 'encoder':
                new_key = '.'.join(key.split('.')[1:])
                state_dict[new_key] = state_dict.pop(key)
                key = new_key
            # remove keys that are not in the encoder
            if key.split('.')[0] in ['incre_modules', 'downsamp_modules', 'final_layer', 'classifier', 'decoder', 'projection_head']:
                del state_dict[key]
        self.encoder.load_state_dict(state_dict)
        

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        h = self.encoder(x)
        
        returns = []
        if self.decoder is not None:
            y = self.decoder(h)
            y = self.img_decoder_activation(y)
            returns.append(y)
        
        if self.projection_head is not None:
            returns.append(self.projection_head(h))
        
        if len(returns) == 1:
            return returns[0]

        return tuple(returns)



class ConvNextTinyAutoencoder(nn.Module):
    
    def __init__(self, 
        img_decoder_head: bool=True,
        use_simple_decoder: bool=True, # if True, use SimpleImageDecoderHead, else use ImageDecoderHead with multiple blocks
        img_decoder_activation: str='sigmoid',
        num_classes: int=3, 
        aux_simclr_head: bool=False,
        pretrained: bool=True
    ):
        super(ConvNextTinyAutoencoder, self).__init__()
        
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = convnext_tiny(weights=weights).features
        
        self.decoder = None
        if img_decoder_head:
            if use_simple_decoder:
                self.decoder = SimpleImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes)
            else:
                self.decoder = ImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes, num_blocks=config['IMAGE_DECODER']['NUM_BLOCKS'])
            
            self.img_decoder_activation = None
            if img_decoder_activation == 'sigmoid':
                self.img_decoder_activation = nn.Sigmoid()
            elif img_decoder_activation == 'softmax':
                self.img_decoder_activation = nn.Softmax(dim=1)
            else:
                self.img_decoder_activation = None
                    
        self.projection_head = None
        if aux_simclr_head:
            self.projection_head = ProjectionHead(in_channels=self.encoder_output_channels)
    
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        reps = []
        for i, layer in self.encoder:
            x = layer(x)
            if (i+1) % 2 == 0:
                reps.append(F.interpolate(x, (x.shape[3], x.shape[4]), mode='bilinear', align_corners=True))
        
        h = torch.cat(reps, 1)
        
        returns = []
        if self.decoder is not None:
            y = self.decoder(h)
            if self.img_decoder_activation is not None:
                y = self.img_decoder_activation(y)
            returns.append(y)
        
        if self.projection_head is not None:
            returns.append(self.projection_head(h))
        
        if len(returns) == 1:
            return returns[0]

        return tuple(returns)



class ResNetAutoencoder(nn.Module):
    
    def __init__(self, 
        img_decoder_head: bool=True,
        use_simple_decoder: bool=True, # if True, use SimpleImageDecoderHead, else use ImageDecoderHead with multiple blocks\
        use_se_decoder: bool=False,
        unet_like_decoder: bool=True,
        img_decoder_activation: str='sigmoid',
        num_classes: int=3, 
        aux_simclr_head: bool=False,
        pretrained: bool=True,
        dropout_rate: float=0.5,
    ):
        super(ResNetAutoencoder, self).__init__()
        
        self.encoder = resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.dropout_rate = dropout_rate
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        
        if unet_like_decoder:
            self.encoder_output_channels = 3840
            def adjusted_forward(self, x: torch.Tensor) -> torch.Tensor:
                
                x_list = []
                # x_dim = x.shape[2:]
                
                for module in self.children():
                    x = module(x)
                    
                    if isinstance(module, nn.Sequential):
                        x_list.append(x)
                
                for i, x in enumerate(x_list):
                    if i == 0:
                        x_dim = x.shape[2:]
                    x_list[i] = F.interpolate(x, x_dim, mode='bilinear', align_corners=True)
                
                return torch.concat(x_list, 1)
        
            self.encoder.forward = functools.partial(adjusted_forward, self.encoder)
            
        else:
            self.encoder_output_channels = 2048
        
        self.final_layer_output_channels = 2048

        self.decoder = None
        if img_decoder_head:
            if use_simple_decoder:
                self.decoder = SimpleImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes)
            elif use_se_decoder:
                self.decoder = SEImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes)
            else:
                self.decoder = ImageDecoderHead(in_channels=self.encoder_output_channels, num_classes=num_classes)
            
            # self.img_decoder_activation = nn.Identity()
            if img_decoder_activation == 'sigmoid':
                self.img_decoder_activation = nn.Sigmoid()
            elif img_decoder_activation == 'softmax':
                self.img_decoder_activation = nn.Softmax(dim=1)
            else:
                self.img_decoder_activation = nn.Identity()
                    
        self.projection_head = None
        if aux_simclr_head:
            self.projection_head = ProjectionHead(in_channels=self.final_layer_output_channels)
    
    
    def load_encoder_weights(self, state_dict: dict):
        for key in list(state_dict.keys()):
            if key.split('.')[0] == 'encoder':
                new_key = '.'.join(key.split('.')[1:])
                state_dict[new_key] = state_dict.pop(key)
                key = new_key
            # remove keys that are not in the encoder
            if key.split('.')[0] in ['incre_modules', 'downsamp_modules', 'final_layer', 'classifier', 'decoder', 'projection_head']:
                del state_dict[key]
        self.encoder.load_state_dict(state_dict)
    
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        h = self.encoder(x)
        
        returns = []
        if self.decoder is not None:
            y = self.decoder(h)
            if self.dropout_rate > 0:
                y = self.dropout(y)
            y = self.img_decoder_activation(y)
            returns.append(y)
        
        if self.projection_head is not None:
            # only pass the output of the last layer of the encoder to the projection head
            h = h[:, -self.final_layer_output_channels:]
            returns.append(self.projection_head(h))
        
        if len(returns) == 1:
            return returns[0]

        return tuple(returns)



class UNetUpBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetUpBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.se = SEBlock(in_channels)
        self.conv_blocks = nn.ModuleList([])
        for i in range(2):
            channels = in_channels if i == 0 else out_channels
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x: torch.Tensor, x_enc: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        x = self.up(x)
        # x = self.se(x)
        if x_enc is not None:
            x = torch.cat([x, x_enc], dim=1)
        x = self.conv_blocks[0](x)
        x = self.conv_blocks[1](x) + x
        return x



class UNet(nn.Module):
    
    def __init__(self, 
        num_classes: int=8,
        pretrained: bool=True, 
        activation: nn.Module=nn.Softmax(dim=1),
        use_extended_decoder: bool=False,
        auxillary_simclr_head: bool=False
    ):
        super(UNet, self).__init__()
        
        self.pretrained = pretrained
        weights = ResNet152_Weights.DEFAULT if pretrained else None
        self.num_classes = num_classes
        self.use_extended_decoder = use_extended_decoder
        self.auxillary_simclr_head = auxillary_simclr_head
        
        self.encoder = resnet152(weights=weights)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.relu,
            ),
            nn.Sequential(
                self.encoder.maxpool,
                self.encoder.layer1
            ),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ])
        
        self.decoder_blocks = nn.ModuleList([
            UNetUpBlock(3072, 1024),
            UNetUpBlock(1536, 512),
            UNetUpBlock(768, 256),
            UNetUpBlock(320, 128),
            UNetUpBlock(128, 64),
        ])
        if self.use_extended_decoder:
            for _ in range(2):
                self.decoder_blocks.append(nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ))
        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.activation = activation
        
        self.projection_head = None
        if auxillary_simclr_head:
            self.projection_head = ProjectionHead(in_channels=2048)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # encoder blocks
        x_list = []
        for block in self.encoder_blocks:
            x = block(x)
            x_list.append(x)
        # x_list = [
        #     stem_features,                (B, 64, 128, 128)
        #     residual_block_1_features,    (B, 256, 64, 64)
        #     residual_block_2_features,    (B, 512, 32, 32)
        #     residual_block_3_features,    (B, 1024, 16, 16)
        #     residual_block_4_features,    (B, 2048, 8, 8)
        # ]
        
        x = self.decoder_blocks[0](x_list[-1], x_list[-2]) # (B, 1024, 16, 16)
        x = self.decoder_blocks[1](x, x_list[-3])          # (B, 512, 32, 32)
        x = self.decoder_blocks[2](x, x_list[-4])          # (B, 256, 64, 64)
        x = self.decoder_blocks[3](x, x_list[-5])          # (B, 128, 128, 128)
        
        # no concatenation with encoder features for last block, just bringing features back up to original size
        x_1 = self.decoder_blocks[4](x)                    # (B, 64, 256, 256)
        
        if self.use_extended_decoder:
            # final decoder blocks are just basic convolutional blocks
            x = self.decoder_blocks[5](x_1) + x_1          # (B, 64, 256, 256) 
            x = self.decoder_blocks[6](x) + x_1 + x        # (B, 64, 256, 256)
        else:
            x = x_1
        
        x = self.classifier(x)                             # (B, num_classes, 256, 256)
        x = self.activation(x)
        
        if self.projection_head is not None:
            return x, self.projection_head(x_list[-1])
        
        return x

    
    
    def load_encoder_weights(self, state_dict: dict):
        for key in list(state_dict.keys()):
            if key.split('.')[0] == 'encoder':
                new_key = '.'.join(key.split('.')[1:])
                state_dict[new_key] = state_dict.pop(key)
                key = new_key
            # remove keys that are not in the encoder
            if key.split('.')[0] in ['incre_modules', 'downsamp_modules', 'final_layer', 'classifier', 'decoder', 'projection_head']:
                del state_dict[key]
        self.encoder.load_state_dict(state_dict)
    
    
    
    def freeze_encoder(self):
        for encoder_block in self.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = False
    
    
    
    def unfreeze_encoder(self):
        for encoder_block in self.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = True
    
    
    
    def freeze_decoder(self):
        for decoder_block in self.decoder_blocks:
            for param in decoder_block.parameters():
                param.requires_grad = False
    
    
    
    def unfreeze_decoder(self):
        for decoder_block in self.decoder_blocks:
            for param in decoder_block.parameters():
                param.requires_grad = True



# class UConvNeXT(UNet, nn.Module):
    
#     def __init__(self, 
#         num_classes: int=8,
#         pretrained: bool=True, 
#         activation: nn.Module=nn.Softmax(dim=1),
#         use_extended_decoder: bool=True,
#         auxillary_simclr_head: bool=False
#     ):
#         super(UConvNeXT, self).__init__(num_classes=num_classes, pretrained=pretrained, activation=activation, use_extended_decoder=use_extended_decoder, auxillary_simclr_head=auxillary_simclr_head)
        
#         self.encoder = convnext_tiny(weights=ResNet152_Weights.DEFAULT if pretrained else None).features
