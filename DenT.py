import copy
from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class conv_block_nested(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class Unet_Block(nn.Module):
    '''Construct the Unet block'''
    def __init__(self):
        super().__init__()
        
        self.n_in = 3
        
        n1 = 32
        filters = [n1, n1*2, n1*4, n1*8, n1*16]
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv0 = conv_block_nested(self.n_in, filters[0])
        self.conv1 = conv_block_nested(filters[0], filters[1])
        self.conv2 = conv_block_nested(filters[1], filters[2])
        self.conv3 = conv_block_nested(filters[2], filters[3])
        self.conv4 = conv_block_nested(filters[3], filters[4])
    
    def forward(self, x):
        features = []
        x = self.conv0(x)
        features.append(x)

        x = self.conv1(self.pool(x))
        features.append(x)
        
        x = self.conv2(self.pool(x))
        features.append(x)
        
        x = self.conv3(self.pool(x))
        features.append(x)
        
        x = self.conv4(self.pool(x))
        features.append(x)
        
        return features[::-1]


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, add_pos_emb=True):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        filters = [16,8,4,2]
        self._add_pos_emb = False

        patch_size = []
        for i in range(len(filters)):
            scale = filters[i]
            grid_size = [2,16,16]
            patch_size.append([img_size[0] // scale // grid_size[0], img_size[1] // scale // grid_size[1], img_size[2] // scale // grid_size[2]])
        n_patches = 512 #2*16*16  

        self.hybrid_model = Unet_Block()
        in_channels = [512, 256, 128, 64]
        out_channels = [768, 384, 192, 96]

        patch_embedding_blocks = [
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size) for in_channels, out_channels, patch_size in zip(in_channels, out_channels, patch_size)
        ]
        self.patch_embeddings = nn.ModuleList(patch_embedding_blocks)

        if add_pos_emb:
            self.position_embeddings1 = nn.Parameter(torch.zeros(1, n_patches, out_channels[0]))
            self.position_embeddings2 = nn.Parameter(torch.zeros(1, n_patches, out_channels[1]))
            self.position_embeddings3 = nn.Parameter(torch.zeros(1, n_patches, out_channels[2]))
            self.position_embeddings4 = nn.Parameter(torch.zeros(1, n_patches, out_channels[3]))
            self._add_pos_emb = True
       
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        embeddings = []
        features = self.hybrid_model(x)

        for i, patch_block in enumerate(self.patch_embeddings):
            embeddings.append(patch_block(features[i])) # (B, hidden, n_patches^(1/2), n_patches^(1/2), n_patches^(1/2))
            embeddings[i] = embeddings[i].flatten(2) # (B, hidden, DWH/P^3)
            embeddings[i] = embeddings[i].transpose(-1, -2) # (B, n_patches, hidden)
            if not self._add_pos_emb:
                embeddings[i] = self.dropout(embeddings[i])
        
        if self._add_pos_emb:
            embeddings[0] = self.dropout(embeddings[0] + self.position_embeddings1)
            embeddings[1] = self.dropout(embeddings[1] + self.position_embeddings2)
            embeddings[2] = self.dropout(embeddings[2] + self.position_embeddings3)
            embeddings[3] = self.dropout(embeddings[3] + self.position_embeddings4)

        return embeddings, features[1:]


class Mlp(nn.Module):
    def __init__(self, out_channels):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(out_channels, out_channels*4)
        self.fc2 = nn.Linear(out_channels*4, out_channels)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, vis, out_channels):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.attn = nn.MultiheadAttention(out_channels, 12, batch_first=True)
        self.ffn = Mlp(out_channels)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, vis, out_channels):
        super(Encoder, self).__init__()
        self.vis = vis
        self.out_channels = out_channels
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(out_channels, eps=1e-6)
        for _ in range(12):
            layer = Block(vis, out_channels)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        
        out_channels = [768, 384, 192, 96]
        encoder_blocks = [Encoder(vis, o_ch) for o_ch in out_channels]
        self.encoder = nn.ModuleList(encoder_blocks)

    def forward(self, input_ids):
        encoded = []
        attn_weights = []
        embedding_output, features = self.embeddings(input_ids)
        for i, encoder in enumerate(self.encoder):
            x, y = encoder(embedding_output[i]) # (B, n_patch, hidden)
            encoded.append(x)
            attn_weights.append(y)
            
        return encoded, attn_weights, features


class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        head_channels = [512, 256, 128, 64]
        out_channels = [768, 384, 192, 96]
        
        conv = [
            nn.Sequential(
                nn.Conv3d(out_channels, head_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(head_channels),
                nn.ReLU(inplace=True),
                ) for out_channels, head_channels in zip(out_channels, head_channels)
        ]
        self.conv_more = nn.ModuleList(conv)

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv_embedding1_0 = conv_block_nested(256,128)

        self.conv_embedding2_0 = conv_block_nested(128,64)
        self.conv_embedding2_1 = conv_block_nested(64,32)
        
        self.conv_embedding3_0 = conv_block_nested(64,32)
        self.conv_embedding3_1 = conv_block_nested(32,16)
        self.conv_embedding3_2 = conv_block_nested(16,8)
        
        self.conv3_1 = conv_block_nested(256+512, 256)
        self.conv2_1 = conv_block_nested(128+128, 128)
        self.conv1_1 = conv_block_nested(64+32, 64)
        self.conv0_1 = conv_block_nested(32+8, 32)
        self.conv2_2 = conv_block_nested(128+128+256, 128)
        self.conv1_2 = conv_block_nested(64+64+128, 64)
        self.conv0_2 = conv_block_nested(32+32+64, 32)
        self.conv1_3 = conv_block_nested(64+64+64+128, 64)
        self.conv0_3 = conv_block_nested(32+32+32+64, 32)
        self.conv0_4 = conv_block_nested(32+32+32+32+64, 32)
        
        self.final = conv_block_nested(32, 16)

    def forward(self, hidden_states, features=None):
        states = []
        channels = [768, 384, 192, 96]
        B, n_patch, _ = hidden_states[0].size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        d, h, w = 2, int(np.sqrt(n_patch/2)), int(np.sqrt(n_patch/2))
        
        for i, convolution in enumerate(self.conv_more):
            x = hidden_states[i].permute(0, 2, 1)
            x = x.contiguous().view(B, channels[i], d, h, w)
            x = convolution(x)
            states.append(x)
        
        states[1] = self.conv_embedding1_0(self.up(states[1]))
        states[2] = self.conv_embedding2_0(self.up(states[2]))
        states[2] = self.conv_embedding2_1(self.up(states[2]))
        states[3] = self.conv_embedding3_0(self.up(states[3]))
        states[3] = self.conv_embedding3_1(self.up(states[3]))
        states[3] = self.conv_embedding3_2(self.up(states[3]))

        x3_1 = self.conv3_1(torch.cat([features[0], self.up(states[0])], dim=1))
        x2_1 = self.conv2_1(torch.cat([features[1], self.up(states[1])], dim=1))
        x1_1 = self.conv1_1(torch.cat([features[2], self.up(states[2])], dim=1))
        x0_1 = self.conv0_1(torch.cat([features[3], self.up(states[3])], dim=1))
        
        x2_2 = self.conv2_2(torch.cat([features[1], x2_1, self.up(x3_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([features[2], x1_1, self.up(x2_1)], dim=1))
        x0_2 = self.conv0_2(torch.cat([features[3], x0_1, self.up(x1_1)], dim=1))

        x1_3 = self.conv1_3(torch.cat([features[2], x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_3 = self.conv0_3(torch.cat([features[3], x0_1, x0_2, self.up(x1_2)], dim=1))
    
        x0_4 = self.conv0_4(torch.cat([features[3], x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))
    
        out = self.final(x0_4)

        return out


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear', align_corners=True) if upsampling > 1 else nn.Identity()
        super().__init__(conv3d, upsampling)


class DenseTransformer(nn.Module):
    def __init__(self, args, img_size=[32,256,256], zero_head=False, vis=False):
        super().__init__() # 
        self.zero_head = zero_head
        self.transformer = Transformer(img_size, vis)
        self.decoder = DecoderCup()
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits


class CustomizableDenT(nn.Module):
    def __init__(self, 
                 img_size=[32,256,256], 
                 out_channels=None, 
                 use_multiheads=None, 
                 return_weights=False,
                 add_pos_emb=True):
        super().__init__()
        
        if out_channels is None:
            out_channels = [768, 384, 192, 96]

        if use_multiheads is None:
            use_multiheads = [True for _ in range(len(out_channels))]

        assert len(out_channels) == len(use_multiheads), f"The length of out_channels ({len(out_channels)}) must matches the length of use_multiheads ({len(use_multiheads)})"

        self.embeddings = Embeddings(img_size=img_size, add_pos_emb=add_pos_emb)
        encoder_blocks = [Encoder(return_weights, o_ch) if use_multiheads[i] else None
                          for i, o_ch in enumerate(out_channels)]
        self.encoder = nn.ModuleList(encoder_blocks)
        
        self.decoder = DecoderCup()
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
        embedding_output, features = self.embeddings(x)
        encoded = []
        attn_weights = []
        for i, encoder in enumerate(self.encoder):
            if encoder is None:
                encoded.append(embedding_output[i])
                attn_weights.append(None)
                continue
            x, y = encoder(embedding_output[i]) # (B, n_patch, hidden)
            encoded.append(x)
            attn_weights.append(y)

        x = self.decoder(encoded, features)
        logits = self.segmentation_head(x)
        return logits
