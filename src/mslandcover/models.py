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
from timm.models import convnext
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
        
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.hiddens = nn.ModuleList([])
        for _ in range(num_hiddens):
            self.hiddens.append(nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.ReLU(inplace=True)
            ))
        
        self.output = nn.Linear(in_channels, embedding_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
                
        for hidden_layer in self.hiddens:
            x = hidden_layer(x)

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
        img_decoder_activation_fn: nn.Module=nn.Sigmoid(),
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
            
            if img_decoder_activation_fn is None:
                self.img_decoder_activation = nn.Identity()
            else:
                self.img_decoder_activation = img_decoder_activation_fn
                    
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
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int=2):
        super(UNetUpBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_blocks = nn.ModuleList([])
        for i in range(num_blocks):
            channels = in_channels + skip_channels if i == 0 else out_channels
            self.conv_blocks.append(ConvBlock(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation='relu',
            ))
    
    def forward(self, x: torch.Tensor, x_enc: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        x = self.up(x)
        # x = self.se(x)
        if x_enc is not None:
            x = torch.cat([x, x_enc], dim=1)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        return x



class ImprovedUnetUpBlock(nn.Module):
    
    def __init__(self,
        encoder_channels: int,
        decoder_channels: int,
        out_channels: int,
        bilinear_upsample: bool=False,
        n_convs: int=4,
    ):
        super(ImprovedUnetUpBlock, self).__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels
        self.bilinear_upsample = bilinear_upsample
        
        if bilinear_upsample:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(decoder_channels, out_channels, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(decoder_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv_blocks = nn.ModuleList([])
        self.proj = nn.Sequential(
            nn.Conv2d(encoder_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for i in range(n_convs):
            channels = encoder_channels + out_channels if i == 0 else out_channels
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x: torch.Tensor, x_enc: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        x = self.up(x)
        if x_enc is not None:
            x = torch.cat([x, x_enc], dim=1)
        for i, conv_block in enumerate(self.conv_blocks):
            if i == 0:
                x = conv_block(x) + self.proj(x)
            else:
                x = conv_block(x) + x
        return x


class HighResUNet(nn.Module):
    
    def __init__(self, 
        num_classes: int=8,
        pretrained: bool=True, 
        activation: nn.Module=nn.Softmax(dim=1),
        deep_supervision: bool=False
    ):
        super(HighResUNet, self).__init__()
        
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        self.encoder = resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_blocks = nn.ModuleList([
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ])
        
        self.decoder_blocks = nn.ModuleList([
            UNetUpBlock(3072, 1024),
            UNetUpBlock(1536, 512),
            UNetUpBlock(768, 256),
        ])
        
        if self.deep_supervision:
            self.classifiers = nn.ModuleList([
                nn.Conv2d(256, num_classes, kernel_size=1),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.Conv2d(1024, num_classes, kernel_size=1),
                nn.Conv2d(2048, num_classes, kernel_size=1),
            ])
        else:
            self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
    
        self.activation = activation

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = self.stem(x)
        x_enc_feature_maps = []
        for block in self.encoder_blocks:
            x = block(x)
            x_enc_feature_maps.append(x)
        
        if self.deep_supervision:
            x_dec_feature_maps = [x_enc_feature_maps[-1]]
            # x_dec_feature_maps.append(x_enc_feature_maps[-1])
            
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, x_enc_feature_maps[-(i+2)])
            
            if self.deep_supervision:
                x_dec_feature_maps.append(x)
        
        if self.deep_supervision:
            y_out = []
            for i, classifier in enumerate(self.classifiers):
                y = classifier(x_dec_feature_maps[-(i+1)])
                y = self.activation(y)
                y_out.append(y)
            return tuple(y_out)
        
        y = self.classifier(x)
        return self.activation(y)
    
    
    
    def freeze_encoder(self, freeze_stem: bool=True) -> None:
        for encoder_block in self.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = False
        if freeze_stem:
            for param in self.stem.parameters():
                param.requires_grad = False
    
    
    
    def freeze_decoder(self) -> None:
        for decoder_block in self.decoder_blocks:
            for param in decoder_block.parameters():
                param.requires_grad = False
    
    
    
    def reinit_classifier(self, num_classes: int=8) -> None:
        if self.deep_supervision:
            self.classifiers = nn.ModuleList([
                nn.Conv2d(256, num_classes, kernel_size=1),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.Conv2d(1024, num_classes, kernel_size=1),
                nn.Conv2d(2048, num_classes, kernel_size=1),
            ])
        else:
            self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        self.num_classes = num_classes
        return self
    
    
    def disable_deep_supervision(self) -> None:
        self.deep_supervision = False
        self.classifiers = None


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



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling along channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)  # Concatenate along channel axis
        out = self.conv(out)
        return x * self.sigmoid(out)



class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x



class ConvBlock(nn.Module):
    
    """A simple convolutional block followed by batch normalization and ReLU activation."""
    def __init__(self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        batch_norm=True, 
        activation='relu',
        cbam_kernel_size=7,
        cba_reduction=16,
        enable_cbam=False,
    ):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.activation = activation
        self.enable_cbam = enable_cbam
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if enable_cbam:
            self.cbam = CBAM(out_channels, reduction=cba_reduction, kernel_size=cbam_kernel_size)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
            
        if activation is not None:
            if activation == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'leaky_relu':
                self.act = nn.LeakyReLU(inplace=True)
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()
            elif activation == 'softmax':
                self.act = nn.Softmax(dim=1)
            elif activation == 'tanh':
                self.act = nn.Tanh()
            else:
                raise ValueError(f'Invalid value for `activation`: {activation}. Supported values are ["relu", "leaky_relu", "sigmoid", "softmax", "tanh"].')
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        
        if self.activation is not None:
            x = self.act(x)
        
        if self.enable_cbam:
            x = self.cbam(x)
        return x



class AttentionGate(nn.Module):
    """
    Attention Gate module for Attention U-Net.
    
    This module implements the attention mechanism that helps the model
    focus on relevant features during the decoding process.
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        """
        Initialize the attention gate.
        
        Args:
            f_g: Number of channels in the gating signal (from the upper level)
            f_l: Number of channels in the input feature map (skip connection)
            f_int: Number of channels for the intermediate representations
        """
        super(AttentionGate, self).__init__()
        
        # Gating signal convolution (signals from the decoder)
        # self.W_g = nn.Sequential(
        #     nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(f_int)
        # )
        self.W_g = ConvBlock(f_g, f_int, kernel_size=1, stride=1, padding=0, batch_norm=True, activation=None)
        
        # Skip connection convolution (signals from the encoder)
        # self.W_x = nn.Sequential(
        #     nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(f_int)
        # )
        self.W_x = ConvBlock(f_l, f_int, kernel_size=1, stride=1, padding=0, batch_norm=True, activation=None)
        
        # Output convolution
        # self.psi = nn.Sequential(
        #     nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )
        self.psi = ConvBlock(f_int, 1, kernel_size=1, stride=1, padding=0, batch_norm=True, activation='sigmoid')
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention gate.
        
        Args:
            g: Gating signal (from the decoder)
            x: Skip connection input (from the encoder)
            
        Returns:
            Attention-weighted feature map
        """
        
        # Get input sizes
        input_size = x.size()
        
        # Downsample x if needed (match the shape of g)
        if x.size(2) > g.size(2):
            # Downsample x to match g's spatial dimensions
            x_down = F.interpolate(x, size=g.size()[2:], mode='bilinear', align_corners=False)
        else:
            x_down = x
        
        # Apply convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x_down)
        
        # Sum the feature maps and apply ReLU
        psi = self.relu(g1 + x1)
        
        # Apply convolution and sigmoid to get attention coefficients
        psi = self.psi(psi)
        
        # Upsample attention map to original size if needed
        if psi.size(2) < input_size[2]:
            psi = F.interpolate(psi, size=input_size[2:], mode='bilinear', align_corners=False)
        
        # Apply attention weights to input feature map
        return x * psi


class AttentionUnetUpBlock(nn.Module):
    """
    Upsampling block for Attention U-Net with attention gates.
    This replaces the standard ImprovedUnetUpBlock with attention mechanism.
    """
    def __init__(self,
        decoder_channels: int,
        encoder_channels: int,
        out_channels: int,
        upsample: bool=True,
        use_cbam: bool=False,
        bilinear_upsample: bool=True,
        n_convs: int=2,
        activation_func='relu',
    ):
        super(AttentionUnetUpBlock, self).__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.use_cbam = use_cbam
        self.bilinear_upsample = bilinear_upsample
        self.convs = n_convs
        self.activation_func = activation_func
        
        if upsample:
            if bilinear_upsample:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    # nn.Conv2d(decoder_channels, out_channels, kernel_size=1)
                    ConvBlock(decoder_channels, out_channels, kernel_size=1, stride=1, padding=0, batch_norm=True, activation=activation_func)
                )
            else:
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(decoder_channels, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    # CBAM(out_channels, reduction=16, kernel_size=7)
                )
            
            # Add attention gate
            if encoder_channels > 0:  # Only add if there's a skip connection
                self.attention_gate = AttentionGate(
                    f_g=decoder_channels,        # Gating signal channels (from the decoder)
                    f_l=encoder_channels,        # Skip connection channels
                    f_int=encoder_channels // 4  # Intermediate representation channels
                )
        
        # self.proj = nn.Conv2d(encoder_channels + out_channels, out_channels, kernel_size=1)
        # self.proj = ConvBlock(encoder_channels + out_channels, out_channels, kernel_size=1, stride=1, padding=0, batch_norm=True, activation=activation_func)
        self.conv_blocks = nn.ModuleList([])
        for i in range(n_convs):
            channels = encoder_channels + out_channels if i == 0 else out_channels
            self.conv_blocks.append(ConvBlock(channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True, activation=activation_func))
        
        if use_cbam:
            self.cbam = CBAM(out_channels)
    
    def forward(self, x: torch.Tensor, x_enc: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the attention upsample block.
        
        Args:
            x: Input feature map from the previous decoder block
            x_enc: Skip connection from the encoder (optional)
            
        Returns:
            Processed feature map
        """
                
        # Upsample the input
        if self.upsample:
            x_up = self.up(x)
        else:
            x_up = x
        
        # Apply attention mechanism if there's a skip connection
        if x_enc is not None:
            # Apply attention gate to focus on relevant features
            x_enc = self.attention_gate(x, x_enc)

            # Concatenate with upsampled features
            x = torch.cat([x_up, x_enc], dim=1)

        else:
            x = x_up

            
        # Apply convolution blocks
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
        
        # Apply CBAM if enabled
        if self.use_cbam:
            x = self.cbam(x) + x
        
        return x


class AttentionUResNetD(nn.Module):
    """
    Attention U-Net with ResNet backbone encoder.
    This replaces the standard UResNetD with attention mechanisms.
    """
    def __init__(self,
        num_classes: int=8,
        pretrained: bool=True,
        activation: nn.Module=nn.Softmax(dim=1),
        deep_supervision: bool=False,
        decoder_convs: int=2,
        bilinear_upsample: bool=True,
    ):
        super(AttentionUResNetD, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.activation = activation
        self.deep_supervision = deep_supervision
        self.decoder_convs = decoder_convs
        self.bilinear_upsample = bilinear_upsample
        
        # Initialize the encoder (ResNet backbone)
        self.encoder = resnet.resnet152d(pretrained=pretrained)
        self.encoder.global_pool = nn.Identity()
        self.encoder.fc = nn.Identity()
        
        # Group encoder layers into blocks
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.act1,
            ),
            nn.Sequential(
                self.encoder.maxpool,
                self.encoder.layer1
            ),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ])
        
        # Initialize decoder blocks with attention
        self.decoder_blocks = nn.ModuleList([
            AttentionUnetUpBlock(1024, 2048, 1024, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample),
            AttentionUnetUpBlock(512, 1024, 512, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample),
            AttentionUnetUpBlock(256, 512, 256, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample),
            AttentionUnetUpBlock(64, 256, 64, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample),
            AttentionUnetUpBlock(0, 64, 64, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample),
        ])
        
        # Initialize the classifiers
        if self.deep_supervision:
            self.classifiers = nn.ModuleList([
                nn.Conv2d(64, num_classes, kernel_size=1),
                nn.Conv2d(64, num_classes, kernel_size=1),
                nn.Conv2d(256, num_classes, kernel_size=1),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.Conv2d(1024, num_classes, kernel_size=1),
            ])
        else:
            self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Attention U-Net.
        
        Args:
            x: Input image tensor
            
        Returns:
            Model output (segmentation map)
        """
        # Encoder path
        x_enc_feature_maps = []
        for block in self.encoder_blocks:
            x = block(x)
            x_enc_feature_maps.append(x)
        
        if self.deep_supervision:
            x_dec_feature_maps = [x_enc_feature_maps[-1]]
        
        # Decoder path with attention mechanisms
        for i, block in enumerate(self.decoder_blocks):
            if i == len(self.decoder_blocks) - 1:
                x = block(x)  # No skip connection for the last block
            else:
                x = block(x, x_enc_feature_maps[-(i+2)])  # With skip connection and attention
            
            if self.deep_supervision:
                x_dec_feature_maps.append(x)
        
        # Output processing
        if self.deep_supervision:
            y_out = []
            for i, classifier in enumerate(self.classifiers):
                y = classifier(x_dec_feature_maps[-(i+1)])
                y = self.activation(y)
                y_out.append(y)
            return tuple(y_out)
        
        y = self.classifier(x)
        return self.activation(y)
    
    def freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for encoder_block in self.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = False
    
    def freeze_decoder(self) -> None:
        """Freeze the decoder parameters."""
        for decoder_block in self.decoder_blocks:
            for param in decoder_block.parameters():
                param.requires_grad = False
    
    def reinit_classifier(self, num_classes: int=8) -> None:
        """Reinitialize the classifier with a new number of classes."""
        if self.deep_supervision:
            self.classifiers = nn.ModuleList([
                nn.Conv2d(64, num_classes, kernel_size=1),
                nn.Conv2d(64, num_classes, kernel_size=1),
                nn.Conv2d(256, num_classes, kernel_size=1),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.Conv2d(1024, num_classes, kernel_size=1),
            ])
        else:
            self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.num_classes = num_classes
        return self
    
    def disable_deep_supervision(self) -> None:
        """Disable deep supervision."""
        self.deep_supervision = False
        self.classifiers = None



# class SegformerForSimCLR(nn.Module):
#     def __init__(self, model_name="nvidia/mit-b5", projection_dim=128):
#         super().__init__()
        
#         # Load the pretrained SegFormer model
#         self.segformer = SegformerModel.from_pretrained(model_name)
#         self.hidden_size = self.segformer.config.hidden_sizes[-1]  # Get the final hidden size
        
#         # Create projection head for SimCLR
#         self.projection_head = ProjectionHead(in_channels=self.hidden_size, embedding_dim=projection_dim)
    
#     def forward(self, x):
#         # Intermediate representation corresponding to the final hidden state
#         z = self.segformer(x).last_hidden_state
        
#         # Apply projection head
#         projected_features = self.projection_head(z)
        
#         return projected_features
    
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in atrous_rates
        ])
        
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels + 2 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        
        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(global_features, size=size, mode='bilinear', align_corners=False)
        
        conv1x1_features = self.conv1x1(x)
        atrous_features = [conv(x) for conv in self.atrous_convs]
        
        all_features = torch.cat([conv1x1_features, *atrous_features, global_features], dim=1)
        return self.conv1x1_out(all_features)



class AttentionUConvNeXt(nn.Module):
    """
    
    """
    def __init__(self,
        num_classes: int=8,
        pretrained: bool=True,
        activation: nn.Module=nn.Softmax(dim=1),
        decoder_convs: int=8,
        bilinear_upsample: bool=True,
    ):
        super(AttentionUConvNeXt, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.decoder_convs = decoder_convs
        self.bilinear_upsample = bilinear_upsample
        
        self.encoder = convnext.convnext_base(pretrained=pretrained)
        self.encoder.head = nn.Identity()  # Remove the classification head
        
        # Group encoder layers into blocks
        self.encoder_blocks = nn.ModuleList([
            self.encoder.stem,
            self.encoder.stages[0],
            self.encoder.stages[1],
            self.encoder.stages[2],
            self.encoder.stages[3],
        ])
        
        self.bottleneck_cbam = CBAM(1024)  # Apply CBAM after the last encoder block
        
        # self.aspp = ASPP(1024, 1024)  # Apply ASPP after the last encoder block
        
        # Initialize decoder blocks with attention
        self.decoder_blocks = nn.ModuleList([
            AttentionUnetUpBlock(512, 1024, 512, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample, use_cbam=True),
            AttentionUnetUpBlock(256, 512, 256, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample, use_cbam=True),
            AttentionUnetUpBlock(128, 256, 128, n_convs=decoder_convs, bilinear_upsample=bilinear_upsample, use_cbam=True),
        ])
        
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        self.activation = activation
        
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Attention U-Net.
        
        Args:
            x: Input image tensor
            
        Returns:
            Model output (segmentation map)
        """
        in_shape = x.shape[2:]
        
        # Encoder path
        x_enc_feature_maps = []
        for block in self.encoder_blocks:
            x = block(x)
            x_enc_feature_maps.append(x)
        
        x = self.bottleneck_cbam(x)
        
        # Decoder path with attention mechanisms
        for i, block in enumerate(self.decoder_blocks):
            # print(x.shape, x_enc_feature_maps[-(i + 2)].shape)
            x = block(x, x_enc_feature_maps[-(i + 2)])
                
        y = self.classifier(x)
        y = self.activation(y)
        y = F.interpolate(y, size=in_shape, mode='bilinear')
        return y
    
    def freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for encoder_block in self.encoder_blocks:
            for param in encoder_block.parameters():
                param.requires_grad = False
    
    def freeze_decoder(self) -> None:
        """Freeze the decoder parameters."""
        for decoder_block in self.decoder_blocks:
            for param in decoder_block.parameters():
                param.requires_grad = False
        
        for param in self.bottleneck_cbam.parameters():
            param.requires_grad = False
            
    
    def reinit_classifier(self, num_classes: int=8) -> None:
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        self.num_classes = num_classes
        return self


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv + pointwise conv."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Depthwise: groups=in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias
        )
        # Pointwise: 1x1 convolution to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)



class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling as in DeepLab v3+."""
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: tuple[int, ...]) -> None:
        super().__init__()
        # 1×1 conv branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # parallel atrous conv branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for rate in dilation_rates
        ])
        # image-level pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # combine & project
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(dilation_rates)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = [self.conv_1x1(x)] + [branch(x) for branch in self.branches]
        # image-level features
        img_feat = self.image_pool(x)
        img_feat = nn.functional.interpolate(img_feat, size=size, mode="bilinear", align_corners=False)
        feats.append(img_feat)
        x = torch.cat(feats, dim=1)
        return self.project(x)



class Decoder(nn.Module):
    """DeepLab v3+ decoder that fuses low- and high-level features."""
    def __init__(self, low_level_in: int, low_level_out: int, num_classes: int) -> None:
        super().__init__()
        # Reduce low-level feature channels to low_level_out (e.g. 48)
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_level_in, low_level_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out),
            nn.ReLU(inplace=True),
        )
        # Two separable conv layers to refine concatenated features
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(low_level_out + 256, 256, kernel_size=3, padding=1),
            DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
        )
        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, low_level_feat: torch.Tensor, high_level_feat: torch.Tensor) -> torch.Tensor:
        # Upsample ASPP output by factor 4
        high = nn.functional.interpolate(high_level_feat, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=False)
        low = self.reduce_low(low_level_feat)
        x = torch.cat([low, high], dim=1)
        x = self.refine(x)
        return self.classifier(x)



class ResNetBackbone(nn.Module):
    """
    Wraps a ResNet-152 to output (low_level_feat, high_level_feat).
    output_stride=16: remove stride in layer4; stride=8: also in layer3.
    """
    def __init__(self, output_stride: int = 16, pretrained: bool = True, in_channels=4) -> None:
        super().__init__()
        if isinstance(pretrained, bool):
            resnet = resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            resnet = resnet152()
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            if pretrained is not None:
                resnet.load_state_dict(load_pth(pretrained), strict=True)
        # Modify strides/dilations for atrous convolution
        if output_stride == 16:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
            for block in resnet.layer4:
                block.conv2.dilation = (2, 2)
                block.conv2.padding = (2, 2)
        elif output_stride == 8:
            for layer in [resnet.layer3, resnet.layer4]:
                layer[0].conv2.stride = (1, 1)
                layer[0].downsample[0].stride = (1, 1)
                for block in layer:
                    block.conv2.dilation = (2 if layer is resnet.layer4 else 4,)*2
                    block.conv2.padding = (2 if layer is resnet.layer4 else 4,)*2
        
        # Keep initial layers
        self.initial = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        # Low-level: output of layer1 (conv2_x)
        self.layer1 = resnet.layer1
        # High-level: output of layer2/3/4
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)
        return low_level, high_level



class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+ for semantic segmentation.
    - backbone: module returning (low_level_feat, high_level_feat)
    - num_classes: # of segmentation classes
    - aspp_rates: dilation rates for ASPP
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        aspp_out: int = 256,
        aspp_rates: tuple[int, ...] = (12, 24, 36),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # ASPP on high-level features
        self.aspp = ASPP(in_channels=2048, out_channels=aspp_out, dilation_rates=aspp_rates)
        # Decoder fusing ASPP and low-level (conv2) features
        self.decoder = Decoder(low_level_in=256, low_level_out=48, num_classes=num_classes)
        
        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_level, high_level = self.backbone(x)
        x = self.aspp(high_level)
        x = self.decoder(low_level, x)
        # Final upsample to input resolution
        x = nn.functional.interpolate(x, size=x.shape[-2]*4, mode="bilinear", align_corners=False)
        return self.activation(x)



class ResNetBackboneUNet(nn.Module):
    def __init__(self, in_channels=4, pretrained=True):
        super(ResNetBackboneUNet, self).__init__()
        resnet = resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # H/2
        self.pool = resnet.maxpool                                           # H/4
        self.layer1 = resnet.layer1                                          # H/4
        self.layer2 = resnet.layer2                                          # H/8
        self.layer3 = resnet.layer3                                          # H/16
        self.layer4 = resnet.layer4                                          # H/32

    def forward(self, x):
        x0 = self.initial(x)            # [B, 64, H/2, W/2]
        x1 = self.pool(x0)              # [B, 64, H/4, W/4]
        x2 = self.layer1(x1)            # [B, 256, H/4, W/4]
        x3 = self.layer2(x2)            # [B, 512, H/8, W/8]
        x4 = self.layer3(x3)            # [B, 1024, H/16, W/16]
        x5 = self.layer4(x4)            # [B, 2048, H/32, W/32]
        
        return [x0, x2, x3, x4, x5]



class UNet(nn.Module):
    def __init__(self, backbone: ResNetBackboneUNet, num_classes: int = 8):
        
        super(UNet, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        self.decoder_blocks = nn.ModuleList([
            UNetUpBlock(2048, 1024, 1024),
            UNetUpBlock(1024, 512, 512),
            UNetUpBlock(512, 256, 256),
            UNetUpBlock(256, 64, 64),
            UNetUpBlock(64, 0, 32), # No skip connection for the last block - only upsampling
        ])
        
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.
        
        Args:
            x: Input image tensor
            
        Returns:
            Model output (segmentation map)
        """
        # in_shape = x.shape[2:]
        
        # Encoder path
        x_enc_feature_maps = self.backbone(x)
        x = x_enc_feature_maps.pop()  # Start with the last feature map
        
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, x_enc_feature_maps.pop() if len(x_enc_feature_maps) > 0 else None) # do not use skip connection for the last block
        
        x = self.classifier(x)
        return self.activation(x)



class AttentionUNet(UNet):
    """
    Attention U-Net with ResNet backbone.
    
    This model uses attention mechanisms in the decoder blocks to focus on relevant features.
    """
    def __init__(self, backbone: ResNetBackboneUNet, num_classes: int = 8):
        
        super(AttentionUNet, self).__init__(backbone, num_classes)
        
        # only change the decoder blocks to use attention
        self.decoder_blocks = nn.ModuleList([
            AttentionUnetUpBlock(2048, 1024, 1024),
            AttentionUnetUpBlock(1024, 512, 512),
            AttentionUnetUpBlock(512, 256, 256),
            AttentionUnetUpBlock(256, 64, 64),
            AttentionUnetUpBlock(64, 0, 32), # No skip connection for the last block - only upsampling
        ])