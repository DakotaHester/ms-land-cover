import functools
import math
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models import ResNet152_Weights, resnet152, ResNet101_Weights, resnet101
from timm.models import convnext
from .utils import load_pth
import copy


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


# SimCLRProjectionHead implementation inspired by official TF code https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/tf2/model.py#L157
# per paper, only use one hidden layer and do not apply a non-linearity to the output embeddings
# z_i = W^{(2)} \sigma(W^{(1)} h_i)
class SimCLRProjectionHead(nn.Module):
    
    def __init__(self, in_channels: int=720, num_hiddens: int=1, embedding_dim: int=128):
        super(SimCLRProjectionHead, self).__init__()
        
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
            self.projection_head = SimCLRProjectionHead(in_channels=self.encoder_output_channels)
    
    
    
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
            self.projection_head = SimCLRProjectionHead(in_channels=self.encoder_output_channels)
    
    
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
            self.projection_head = SimCLRProjectionHead(in_channels=self.final_layer_output_channels)
    
    
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
#         self.projection_head = SimCLRProjectionHead(in_channels=self.hidden_size, embedding_dim=projection_dim)
    
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
    Wraps a ResNet-101 to output (low_level_feat, high_level_feat).
    output_stride=16: remove stride in layer4; stride=8: also in layer3.
    """
    def __init__(self, output_stride: int = 16, pretrained: bool = True, in_channels=4) -> None:
        super().__init__()
        if isinstance(pretrained, bool):
            # resnet = resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
            resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)

            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            # resnet = resnet152()
            resnet = resnet101()
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
        x1 = self.initial(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5



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
        _, low_level, _, _, high_level = self.backbone(x)
        x = self.aspp(high_level)
        x = self.decoder(low_level, x)
        # Final upsample to input resolution
        x = nn.functional.interpolate(x, size=x.shape[-2]*4, mode="bilinear", align_corners=False)
        return self.activation(x)



class ResNetBackboneUNet(nn.Module):
    def __init__(self, in_channels=4, pretrained=True):
        super(ResNetBackboneUNet, self).__init__()
        # resnet = resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)

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



class MultiScaleLinearProbingResNet(nn.Module):
    def __init__(self, backbone, num_classes=8):
        super(MultiScaleLinearProbingResNet, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Conv2d(3904, num_classes, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Identity()

    def forward(self, x):
        input_size = x.shape[-2:]  # Store original input size
        features = self.backbone(x)
        
        # Interpolate all features to match the largest feature map size
        target_size = features[0].shape[-2:]  # Use first (largest) feature map size
        # interpolated_features = [features[0]]  # First feature doesn't need interpolation
        
        for i in range(1, len(features)):
            if i == 0:
                continue
            features[i] = F.interpolate(
                features[i], 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Concatenate all features
        features = torch.cat(features, dim=1)
        
        # Apply classifier
        x = self.classifier(features)
        
        # Interpolate to original input size AFTER classification
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # Apply activation last
        x = self.activation(x)
        return x



class SimpleLinearProbingResNet(nn.Module):
    def __init__(self, backbone, num_classes=8):
        super(SimpleLinearProbingResNet, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Identity()

    def forward(self, x):
        input_size = x.shape[-2:]  # Store original input size
        features = self.backbone(x)[-1]
        
        # # Interpolate all features to match the largest feature map size
        # target_size = features[0].shape[-2:]  # Use first (largest) feature map size
        # # interpolated_features = [features[0]]  # First feature doesn't need interpolation
        
        # for i in range(1, len(features)):
        #     if i == 0:
        #         continue
        #     features[i] = F.interpolate(
        #         features[i], 
        #         size=target_size, 
        #         mode='bilinear', 
        #         align_corners=False
        #     )
        
        # # Concatenate all features
        # features = torch.cat(features, dim=1)
        
        # Apply classifier
        x = self.classifier(features)
        
        # Interpolate to original input size AFTER classification
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # Apply activation last
        x = self.activation(x)
        return x


class BYOLProjectionHead(nn.Module):
    """
    Projection head for BYOL: 2-layer MLP with BatchNorm and ReLU.
    """
    def __init__(self, in_dim=2048, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
    
    def forward(self, x):
        return self.net(x)


class BYOLPredictionHead(nn.Module):
    """
    Prediction head for BYOL: 2-layer MLP with BatchNorm and ReLU (as in the original BYOL paper).
    """
    def __init__(self, in_dim=256, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
    def forward(self, x):
        return self.net(x)



class BYOLWrapper(nn.Module):
    """
    BYOL wrapper for online and target encoders, with prediction head on online branch only.
    Implements adjustable momentum parameter with cosine schedule: τ = 1 - (1 - τ_base) * (cos(πk/K) + 1)/2
    """
    def __init__(self, encoder, proj_in_dim=2048, proj_hidden_dim=4096, proj_out_dim=256, pred_hidden_dim=4096, tau_base=0.996, total_steps=None):
        super().__init__()
        
        self.tau_base = tau_base
        self.total_steps = total_steps
        self.current_step = 0
        
        # Online encoder: encoder -> projection head -> prediction head
        self.online_encoder = nn.Sequential(
            encoder,
            BYOLProjectionHead(in_dim=proj_in_dim, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim),
        )
        self.online_predictor = BYOLPredictionHead(in_dim=proj_out_dim, hidden_dim=pred_hidden_dim, out_dim=proj_out_dim)
        # Target encoder: encoder -> projection head
        self.target_encoder = nn.Sequential(
            copy.deepcopy(encoder),
            BYOLProjectionHead(in_dim=proj_in_dim, hidden_dim=proj_hidden_dim, out_dim=proj_out_dim),
        )
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _get_current_tau(self):
        """Calculate current momentum parameter using cosine schedule."""
        if self.total_steps is None or self.total_steps == 0:
            return self.tau_base
        
        # τ = 1 - (1 - τ_base) * (cos(πk/K) + 1)/2
        k = min(self.current_step, self.total_steps)  # Clamp to total_steps
        cosine_term = (math.cos(math.pi * k / self.total_steps) + 1) / 2
        tau = 1 - (1 - self.tau_base) * cosine_term
        return tau

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder parameters with current momentum."""
        tau = self._get_current_tau()
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = target.data * tau + online.data * (1.0 - tau)
        self.current_step += 1

    def forward(self, v, v_prime):
        # Online branch: encoder -> projection -> prediction
        q = self.online_predictor(self.online_encoder(v))
        q_prime = self.online_predictor(self.online_encoder(v_prime))
        # Target branch: encoder -> projection (no prediction head)
        with torch.no_grad():
            z = self.target_encoder(v)
            z_prime = self.target_encoder(v_prime) 
        return q, q_prime, z, z_prime



class MoCoV2ProjectionHead(nn.Module):
    """
    Projection head for MoCo v2: 2-layer MLP.
    Structure: Linear -> ReLU -> Linear (Hidden dim 2048, Out dim 128)
    Reference: 'Improved Baselines with Momentum Contrastive Learning'
    """
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.net(x)

class MoCoWrapper(nn.Module):
    """
    MoCo v2 Wrapper with Single-GPU support.
    """
    def __init__(self, encoder, dim=128, K=65536, m=0.999, T=0.2, mlp_dim=2048, single_gpu=True):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.single_gpu = single_gpu

        # Query encoder
        self.encoder_q = nn.Sequential(
            encoder,
            MoCoV2ProjectionHead(in_dim=mlp_dim, hidden_dim=mlp_dim, out_dim=dim)
        )
        # Key encoder
        self.encoder_k = nn.Sequential(
            copy.deepcopy(encoder),
            MoCoV2ProjectionHead(in_dim=mlp_dim, hidden_dim=mlp_dim, out_dim=dim)
        )

        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_key_encoder(self):
        """
        Momentum update of the key encoder parameters AND buffers.
        Required when using eval() on Key Encoder to avoid distribution mismatch.
        """
        # 1. Update Parameters (Weights/Biases)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        # 2. Update Buffers (Running Mean/Variance)
        for (name_q, buffer_q), (name_k, buffer_k) in zip(
            self.encoder_q.named_buffers(), self.encoder_k.named_buffers()
        ):
            if name_q != name_k:
                raise ValueError(f"Buffer name mismatch: {name_q} vs {name_k}")
            
            # CRITICAL FIX: Skip 'num_batches_tracked' or other integer buffers
            if "num_batches_tracked" in name_q:
                buffer_k.data = buffer_q.data
                continue
            
            if buffer_q.dtype not in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                # Skip non-float buffers
                buffer_k.data = buffer_q.data
                continue

            # Use the same momentum coefficient for stats as for weights
            buffer_k.data = buffer_k.data * self.m + buffer_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # Gather keys from all GPUs if DDP, otherwise just use local keys
        if torch.distributed.is_initialized():
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr
        # Note: If batch_size doesn't divide K perfectly, this acts as a ring buffer
        # and might overwrite slightly unevenly at the wrap-around, which is acceptable.
        if self.K % batch_size != 0:
             # Handle edge case where batch size changes (e.g. last batch)
             # Ideally, drop_last=True in DataLoader prevents this.
             pass

        # We assume K is divisible by batch_size for simplicity in this snippet
        # or that the user uses drop_last=True
        
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Handle wrap-around
            tail = self.K - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue[:, :batch_size-tail] = keys[tail:].T
            
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only works correctly in DistributedDataParallel (DDP) ***
        """
        if not torch.distributed.is_initialized():
            return x, None

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        if not torch.distributed.is_initialized() or idx_unshuffle is None:
            return x

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        # 1. Compute Query Features
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # 2. Compute Key Features
        with torch.no_grad():
            # self._momentum_update_key_encoder()

            # SINGLE GPU FIX: Force eval mode to use running stats and prevent leakage
            if self.single_gpu:
                self.encoder_k.eval()
                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)
            
            # MULTI GPU STANDARD: Use Shuffling BN
            else:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # 3. Compute Logits (Einstein Summation)
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        if self.training: # only update the queue during training
            self._dequeue_and_enqueue(k)

        return logits, labels


# ------------------ UPerNet ------------------
class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        ppm_outs = [x] + [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        x = torch.cat(ppm_outs, dim=1)
        return self.bottleneck(x)

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)) for _ in in_channels_list])

    def forward(self, inputs):
        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False)
        return [fpn_conv(l) for fpn_conv, l in zip(self.fpn_convs, laterals)]

class UPerNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.ppm = PPM(2048, 512)
        self.fpn = FPN([256, 512, 1024, 512], 256)
        self.head = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Identity()

    def forward(self, x):
        feats = self.backbone(x)
        ppm_out = self.ppm(feats[-1])
        fpn_feats = self.fpn([feats[1], feats[2], feats[3], ppm_out])
        # size = x.shape[-2:]
        # instead of using x.shape[-2:], we use the size of the largest feature map
        size = fpn_feats[0].shape[-2:]
        out = torch.cat([F.interpolate(f, size=size, mode='bilinear', align_corners=False) for f in fpn_feats], dim=1)
        out = self.head(out) # now interpolate to the original input size (save some VRAM, should be pretty much the same numerically)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.activation(out)


# ------------------ PSPNet ------------------
class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [x] + [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        return self.bottleneck(torch.cat(priors, dim=1))

class PSPNet(nn.Module):
    def __init__(self, backbone, num_classes, aux_out=False):
        super().__init__()
        self.backbone = backbone
        self.psp = PSPModule(2048)
        self.classifier = nn.Conv2d(2048, num_classes, 1)
        self.aux_out = aux_out
        if aux_out:
            self.aux = nn.Conv2d(1024, num_classes, 1)
        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Identity()

    def forward(self, x):
        if self.aux_out:
            feats = self.backbone(x)
            x_psp = self.psp(feats[-1])
            x_cls = self.classifier(x_psp)
            aux_out = self.aux(feats[-2])
            x_cls = F.interpolate(x_cls, size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux_out = F.interpolate(aux_out, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return self.activation(x_cls), aux_out
        
        else:
            feats = self.backbone(x)[-1]
            x_psp = self.psp(feats)
            x_cls = self.classifier(x_psp)
            x_cls = F.interpolate(x_cls, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return self.activation(x_cls)


# ------------------ BiSeNet ------------------
class AttentionRefinementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, sp, cp):
        feat = torch.cat([sp, cp], dim=1)
        feat = self.convblk(feat)
        attn = self.attn(feat)
        return feat * attn + feat

class BiSeNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # self.aux_out = aux_out
        
        n_bands = backbone.n_bands if hasattr(backbone, 'n_bands') else 3
        
        self.spatial = nn.Sequential(
            nn.Conv2d(n_bands, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.ffm = FeatureFusionModule(2048 + 1024 + 2048 + 256, 256)  # Concatenate and reduce channels
        self.arm1 = AttentionRefinementModule(1024)
        self.arm2 = AttentionRefinementModule(2048)
        self.head = nn.Conv2d(256, num_classes, 1)
        # self.aux1 = nn.Conv2d(1024, num_classes, 1)
        # self.aux2 = nn.Conv2d(2048, num_classes, 1)

        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Identity()

    def forward(self, x):
        feats = self.backbone(x)
        global_feats = self.gap(feats[-1])  # Global Average Pooling on last feature map
        sp = self.spatial(x)
        cp1, cp2 = feats[-2], feats[-1]  # cp1: 1024, cp2: 2048 channels
        cp1 = self.arm1(cp1)  # Apply attention refinement
        cp2 = self.arm2(cp2)  # Apply attention refinement

        # Upsample context features to match spatial path
        feats_up = F.interpolate(global_feats, size=sp.shape[-2:], mode='bilinear', align_corners=False)
        cp1_up = F.interpolate(cp1, size=sp.shape[-2:], mode='bilinear', align_corners=False)
        cp2_up = F.interpolate(cp2, size=sp.shape[-2:], mode='bilinear', align_corners=False)
        cp_final = torch.cat([feats_up, cp1_up, cp2_up], dim=1)  # Concatenate upsampled features
        
        ffm_out = self.ffm(sp, cp_final)
        out = self.head(ffm_out)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # if self.aux_out:
        #     aux1_out = F.interpolate(self.aux1(cp1), size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     aux2_out = F.interpolate(self.aux2(cp2), size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     return out, aux1_out, aux2_out
        # else:
        return self.activation(out)


# ------------------ DANet ------------------
class PAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inter_channels = in_channels // 8
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy / math.sqrt(self.inter_channels), dim=-1)
        value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return torch.tanh(self.gamma) * out + x

class CAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy / math.sqrt(proj_key.size(-1)), dim=-1)
        proj_value = x.view(B, C, -1)
        out = torch.bmm(attention, proj_value).view(B, C, H, W)
        return torch.tanh(self.gamma) * out + x

class DANet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.pam = PAM(2048)
        self.cam = CAM(2048)
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1)
        )
        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Identity()

    def forward(self, x):
        feat = self.backbone(x)[-1]
        pam_out = self.pam(feat)
        cam_out = self.cam(feat)
        out = pam_out + cam_out
        out = self.head(out)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.activation(out)


# ------------------ PAN ------------------
class FPAModule(nn.Module):
    """
    Feature Pyramid Attention Module, the core of PAN.
    This is the corrected implementation based on Figure 3(b) from the paper.
    """
    def __init__(self, in_channels, mid_channels=512):
        super().__init__()
        
        # --- Main Path ---
        # 1x1 conv to reduce channels for the main feature path
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)

        # --- Pyramid Attention Path ---
        # These convolutions with varying kernel sizes create the feature pyramid
        self.pyramid_conv1 = ConvBlock(in_channels, mid_channels, kernel_size=7, padding=3)
        self.pyramid_conv2 = ConvBlock(in_channels, mid_channels, kernel_size=5, padding=2)
        self.pyramid_conv3 = ConvBlock(in_channels, mid_channels, kernel_size=3, padding=1)
        
        # This 1x1 conv creates the final attention map from the fused pyramid features
        self.attention_conv = ConvBlock(mid_channels, 1, kernel_size=1, padding=0)

        # --- Global Pooling Branch ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h, w = x.shape[-2:]

        # --- Main Path ---
        main_features = self.conv1(x)

        # --- Pyramid Attention Path ---
        # CORRECTION 1: Using convolutions with different kernel sizes, not pooling
        p1 = self.pyramid_conv1(x)
        p2 = self.pyramid_conv2(x)
        p3 = self.pyramid_conv3(x)
        
        # CORRECTION 2: Fusing pyramid features via multiplication (step-by-step attention)
        p_fused = p1 + p2 + p3 # Simplified fusion for clarity, paper's is more complex
        
        # Create the spatial attention map
        attention_map = torch.sigmoid(self.attention_conv(p_fused))

        # CORRECTION 3: Applying attention via pixel-wise multiplication
        attended_main_features = main_features * attention_map

        # --- Global Pooling Branch ---
        global_features = self.global_conv(self.global_pool(x))
        global_features = F.interpolate(global_features, size=(h, w), mode='bilinear', align_corners=False)

        # --- Final Combination ---
        # CORRECTION 4: Adding the global context branch to the attended features
        return attended_main_features + global_features


class GAUModule(nn.Module):
    """
    Global Attention Upsample Module, the decoder block of PAN.
    This is the corrected implementation based on Figure 4 from the paper.
    """
    def __init__(self, low_channels, high_channels):
        super().__init__()
        
        # Branch for processing high-level features to get global context
        self.conv_high = ConvBlock(high_channels, high_channels, kernel_size=1, padding=0)
        
        # Branch for processing low-level features
        self.conv_low = ConvBlock(low_channels, high_channels, kernel_size=3, padding=1)

    def forward(self, low_features, high_features):
        # CORRECTION 1: Use Global Average Pooling on high-level features for context
        global_context = F.adaptive_avg_pool2d(high_features, 1)
        
        # Get channel-wise attention vector from global context
        attention_vector = self.conv_high(global_context)
        
        # Process low-level features
        low_processed = self.conv_low(low_features)

        # CORRECTION 2: Apply channel-wise attention to low-level features
        attention_weighted_low = low_processed * attention_vector

        # CORRECTION 3: Add original high-level features to the weighted low-level features
        # The calling function is responsible for upsampling `high_features` to match `low_features` size
        return high_features + attention_weighted_low


class PAN(nn.Module):
    """
    The full Pyramid Attention Network.
    This version uses a proper multi-stage decoder as shown in the paper's Figure 2.
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        # NOTE: This assumes backbone returns a list of features from its stages.
        # Example channel sizes for a ResNet-101 are used.
        # [c2, c3, c4, c5] -> [256, 512, 1024, 2048]
        self.backbone = backbone
        
        # FPA module applied to the last feature map (c5)
        self.fpa = FPAModule(in_channels=2048)
        
        # CORRECTION: A cascade of GAU modules for a multi-stage decoder
        self.gau3 = GAUModule(low_channels=1024, high_channels=512)
        self.gau2 = GAUModule(low_channels=512, high_channels=512)
        self.gau1 = GAUModule(low_channels=256, high_channels=512)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Identity()

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Get features from the backbone encoder
        # This assumes the backbone is set up to return features from stages 2, 3, 4, and 5
        _, c2, c3, c4, c5 = self.backbone(x)

        # 1. Apply FPA to the deepest features
        fpa_out = self.fpa(c5)

        # 2. Start the decoder cascade from top to bottom
        # CORRECTION: Multi-stage decoder path
        
        # Upsample FPA output and fuse with c4 features using GAU
        g3_high = F.interpolate(fpa_out, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        g3_out = self.gau3(c4, g3_high)

        # Upsample previous output and fuse with c3 features
        g2_high = F.interpolate(g3_out, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        g2_out = self.gau2(c3, g2_high)

        # Upsample previous output and fuse with c2 features
        g1_high = F.interpolate(g2_out, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        g1_out = self.gau1(c2, g1_high)

        # 3. Final prediction
        out = self.final_conv(g1_out)
        
        # Upsample to original image size for final output
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return self.activation(out)
    
class FCN(nn.Module):
    """
    Faithful FCN-8s implementation using a ResNet-101 backbone.

    References
    ----------
    Long, Shelhamer, and Darrell. "Fully Convolutional Networks
    for Semantic Segmentation." CVPR 2015 (arXiv:1411.4038v2).

    Notes
    -----
    - Uses skip connections from strides 32, 16, and 8 feature maps.
    - Performs elementwise summation of score maps (no refinement convs).
    - Uses bilinear interpolation for upsampling (fixed, not learned).
    - Returns probabilities (Sigmoid/Softmax) since the loss expects them.
    """
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # 1×1 convs to project backbone feature maps to class scores
        self.score_final = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(512, num_classes, kernel_size=1)

        # activation (not in paper, but required for your setup)
        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]
        feats = self.backbone(x)

        # Expect final three feature maps correspond to strides [8,16,32]
        pool3 = feats[-3]  # stride 8
        pool4 = feats[-2]  # stride 16
        final = feats[-1]  # stride 32

        # score from deepest layer (stride 32)
        score_final = self.score_final(final)

        # upsample x2 and fuse with pool4 (stride 16)
        score_final_up = F.interpolate(score_final, size=pool4.shape[-2:], mode="bilinear", align_corners=False)
        score_pool4 = self.score_pool4(pool4)
        fuse16 = score_final_up + score_pool4

        # upsample x2 and fuse with pool3 (stride 8)
        fuse16_up = F.interpolate(fuse16, size=pool3.shape[-2:], mode="bilinear", align_corners=False)
        score_pool3 = self.score_pool3(pool3)
        fuse8 = fuse16_up + score_pool3

        # final upsample to input resolution
        out = F.interpolate(fuse8, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return self.activation(out)



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Only works correctly in DistributedDataParallel (DDP) ***
    """
    if not torch.distributed.is_initialized():
        return tensor
        
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



class DINOProjectionHead(nn.Module):
    """
    DINO Projection Head.
    Structure: MLP -> L2 Norm -> Weight Normalized Linear -> Softmax (in loss)
    """
    def __init__(self, in_dim, out_dim=65536, use_bn=False, norm_last_layer=True, n_layers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        n_layers = max(n_layers, 1)
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1) # incompatible with parame
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOWrapper(nn.Module):
    """
    DINO wrapper.
    Encapsulates Student, Teacher, Center, and Momentum Schedule (0.996 -> 1.0).
    """
    def __init__(self, encoder, in_dim=2048, out_dim=65536, center_momentum=0.9, 
                 teacher_momentum=0.996, total_steps=None):
        super().__init__()
        
        self.center_momentum = center_momentum
        self.teacher_momentum_base = teacher_momentum
        self.total_steps = total_steps
        self.current_step = 0
        
        # Student
        self.student = nn.Sequential(
            encoder,
            DINOProjectionHead(in_dim, out_dim=out_dim)
        )
        # Teacher
        self.teacher = nn.Sequential(
            copy.deepcopy(encoder),
            DINOProjectionHead(in_dim, out_dim=out_dim)
        )
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Center Buffer
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def _get_current_momentum(self):
        """
        Calculate teacher momentum: Cosine schedule 0.996 -> 1.0.
        Ref: [cite: 281]
        """
        if self.total_steps is None or self.total_steps == 0:
            return self.teacher_momentum_base
            
        # m = 1 - (1 - m_base) * (cos(pi * k / K) + 1) / 2
        # Note: We invert the cosine decay because we want m to INCREASE to 1.0
        k = min(self.current_step, self.total_steps)
        cosine_term = (math.cos(math.pi * k / self.total_steps) + 1) / 2
        m = 1 - (1 - self.teacher_momentum_base) * cosine_term
        return m

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher parameters."""
        m = self._get_current_momentum()
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)
        self.current_step += 1

    @torch.no_grad()
    def update_center(self, teacher_output):
        """EMA update of center."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center = batch_center / torch.distributed.get_world_size()
        batch_center = batch_center / len(teacher_output)
        
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, views):
        # 1. Group inputs (Assumes [global, global, local...])
        global_views = views[:2]
        local_views = views[2:]
        
        # 2. Student Forward (All Crops)
        student_global = [self.student(v) for v in global_views]
        student_local = [self.student(v) for v in local_views]
        student_out = student_global + student_local
        
        # 3. Teacher Forward (Global Only)
        with torch.no_grad():
            teacher_out = [self.teacher(v) for v in global_views]
            
        return student_out, teacher_out