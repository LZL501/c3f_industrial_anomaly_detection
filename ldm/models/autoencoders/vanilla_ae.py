import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from ldm.models.autoencoders.de_resnet import AttnBasicBlock, AttnBottleneck, de_wide_resnet50_2, BN_layer
from ldm.models.autoencoders.losses import FocalLoss

from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention

import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from .quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config


from .utils import FPN, PositionEmbeddingSine, build_position_encoding, nonlinearity, Normalize, make_attn, Backbone_ResNet
from sklearn import metrics
from scipy import ndimage
import cv2

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, with_conv=True, scale=2.0):
        super().__init__()
        self.with_conv = with_conv
        out_channels = in_channels if out_channels is None else out_channels
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.scale = scale

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=1,padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels//2)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class Bottleneck(nn.Module):
    def __init__(self, *, in_channels, mid_channels=None, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        mid_channels = in_channels//2 if mid_channels is None else mid_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1,padding=0)

        self.norm2 = Normalize(mid_channels)
        self.conv2 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.norm3 = Normalize(mid_channels)
        self.conv3 = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.conv2(h)

        h = self.norm3(h)
        h = self.conv3(h)
        h = nonlinearity(h)


        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h




class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), down_samples=(4,2,2,2), num_res_blocks=(2,2,2,2),
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        
        # print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = Bottleneck(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        # self.mid.block_2 = Bottleneck(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(Bottleneck(in_channels=block_in, mid_channels=block_in//2, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level == 1:
                    up.upsample = Upsample(block_in, block_in, resamp_with_conv, scale=down_samples[i_level])
                else:
                    up.upsample = Upsample(block_in, block_in // 2, resamp_with_conv, scale=down_samples[i_level])
                    # up.upsample = Upsample(block_in, block_in, resamp_with_conv, scale=down_samples[i_level])
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, zs):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = zs[-1].shape

        # timestep embedding
        temb = None

        # z to block_in
        # h = self.conv_in(zs[-1])
        h = zs[-1]

        # middle
        # h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)
        ys = []

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # print(h.shape)
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                ys.append(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            if i_level > 1:
                h = torch.cat((h, zs[i_level - 2]), dim=1)
                # h = h + zs[i_level - 2]

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
        # if True:
            h = torch.tanh(h)
        return ys[::-1], h


def dissimilarity(m1, m2):
    return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)


class VQModel(pl.LightningModule):
    def __init__(self,
                 backbone, 
                 ddconfig,
                 lossconfig,
                 n_embeds,
                 feature_dims,
                 pos_dims, 
                 folds, 
                 image_size, 
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Backbone_ResNet(name=backbone, train_backbone=True, return_interm_layers=True, dilation=False)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        
        self.get_pos_embed = nn.ModuleList([PositionEmbeddingSine(pos_dim//2, normalize=True, temperature=10) for pos_dim in pos_dims])
        self.feature_dims = feature_dims
        embed_dims = [(fd + pd) * fold ** 2 for fd, pd, fold in zip(feature_dims, pos_dims, folds)]
        self.shapes = [int(image_size / 2 ** i) for i in range(2, 6)]
        self.folds = folds

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor
        
        self.discriminator = DiscriminativeSubNetwork(in_channels=1, out_channels=2, base_channels=64, out_features=False)
        self.focal_loss = FocalLoss()


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def encode(self, x):
        encoder_features = self.encoder(x)
        return encoder_features
    
    
    def decode(self, quants):
        decoder_features, dec = self.decoder(quants)
        return decoder_features, dec
    

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec


    def forward(self, input):
        encoder_features = self.encode(input)
        decoder_features, dec = self.decode(encoder_features)
     
        return dec, encoder_features, decoder_features


    def get_input(self, batch, k):
        
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format)
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        # x = self.get_input(batch, self.image_key)
        # x, x_anomaly, mask, target = batch
        x = batch
        
        xrec, encoder_features, decoder_features = self(x)
        
        rec_loss = torch.mean(torch.abs(xrec.contiguous() - x.contiguous()))
        
        for i in range(len(encoder_features)):
            rec_loss = rec_loss + 10 * torch.mean(dissimilarity(encoder_features[i].contiguous(), decoder_features[i].contiguous()))
        
        self.log("train/loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return rec_loss

      

        

    def validation_step(self, batch, batch_idx):
        xs, masks, labels = batch
                
        xrec, encoder_features, decoder_features, = self(xs)
        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear')) / 3
       

        preds = np.stack([ndimage.gaussian_filter(p.detach().cpu().numpy(), sigma=4) for p in preds], axis=0)
        res = {"preds": preds, "masks": masks.cpu().numpy(), "labels": labels.cpu().numpy()}

        return res
    
    
    def validation_epoch_end(self, outputs):
        preds = [o["preds"] for o in outputs]
        masks = [o["masks"] for o in outputs]
        labels = [o["labels"] for o in outputs]

        preds = np.concatenate([pred.flatten() for pred in preds], axis=0)
        masks = np.concatenate([mask.flatten() for mask in masks], axis=0)
        masks[masks > 0] = 1
        
        fpr, tpr, thresholds = metrics.roc_curve(masks, preds, pos_label=1)
        auc_pixel = metrics.auc(fpr, tpr)
        self.log("val/pixel_auroc", auc_pixel)
        
        scores = np.concatenate([np.max(o["preds"].reshape(o["preds"].shape[0], -1), axis=1) for o in outputs], axis=0)
        labels = np.concatenate([label.flatten() for label in labels], axis=0)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        auc_image = metrics.auc(fpr, tpr)
        self.log("val/image_auroc", auc_image)
        
        self.log("val/metric", auc_pixel + auc_image)
        
        return auc_pixel + auc_image
    
    
    def test_step(self, batch, batch_idx):
        xs, masks, labels = batch
                
        xrec, encoder_features, decoder_features, = self(xs)
        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear')) / 3
       

        preds = np.stack([ndimage.gaussian_filter(p.detach().cpu().numpy(), sigma=4) for p in preds], axis=0)
        res = {"preds": preds, "masks": masks.cpu().numpy(), "labels": labels.cpu().numpy()}

        return res
    
    
    def test_epoch_end(self, outputs):
        preds = [o["preds"] for o in outputs]
        masks = [o["masks"] for o in outputs]
        labels = [o["labels"] for o in outputs]

        preds = np.concatenate([pred.flatten() for pred in preds], axis=0)
        masks = np.concatenate([mask.flatten() for mask in masks], axis=0)

        masks[masks > 0] = 1
    
        fpr, tpr, thresholds = metrics.roc_curve(masks, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        self.log("val/pixel_auroc", auc)
        
        scores = np.concatenate([np.max(o["preds"].reshape(o["preds"].shape[0], -1), axis=1) for o in outputs], axis=0)
        labels = np.concatenate([label.flatten() for label in labels], axis=0)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        self.log("val/image_auroc", auc)
        
        return auc
    

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        # params = list(self.decoder.parameters())

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    def get_last_layer(self):
        # return None
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        
        def dissimilarity(m1, m2):
            return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)
        
        def normalize(p):
            return ((p - p.min()) / (p.max() - p.min()) * 255).astype(np.uint8)
        
        log = dict()
        if kwargs['split'] == 'train':
            # x = batch[1].to(self.device)
            x = batch.to(self.device)
        else:
            x = batch[0].to(self.device)
            
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        xrec, encoder_features, decoder_features = self(x)
        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear')) / 3
        # preds = self.discriminator(preds, encoder_features, decoder_features)
        
        # xrec = x
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
        x = self.to_rgb(x)  # [B, C, H, W]
        xrec = self.to_rgb(xrec)
        
        result = torch.cat((x, xrec), dim=-2).detach().cpu()
        
        if kwargs['split'] != 'train':
            _, _, h, w = xrec.shape
            # preds = F.softmax(preds, dim=1)[:, 1, :, :].cpu().numpy()
            heatmaps = torch.from_numpy(np.stack([cv2.applyColorMap(cv2.resize(normalize(p[0].cpu().numpy()), (h, w)), cv2.COLORMAP_JET)[:, :, ::-1] for p in preds])).permute(0, 3, 1, 2) / 255 # [B, 3, H, W], 0~1
            heatmaps = (heatmaps * 0.3 + ((xrec.detach().cpu() + 1) / 2).cpu() * 0.5) * 2 - 1
            # result = torch.cat((result, heatmaps, (batch[1] * 2 - 1).repeat(1, 3, 1, 1).detach().cpu()), dim=-2)
            result = torch.cat((result, heatmaps, batch[1].repeat(1, 3, 1, 1).detach().cpu() * 2 - 1), dim=-2)
        # else:
        #     _, _, h, w = xrec.shape
        #     preds = F.softmax(preds, dim=1)[:, 1, :, :].cpu().numpy()
        #     heatmaps = torch.from_numpy(np.stack([cv2.applyColorMap(cv2.resize(normalize(p), (h, w)), cv2.COLORMAP_JET)[:, :, ::-1] for p in preds])).permute(0, 3, 1, 2) / 255 # [B, 3, H, W], 0~1
        #     heatmaps = (heatmaps * 0.3 + ((xrec.detach().cpu() + 1) / 2).cpu() * 0.5) * 2 - 1
        #     # result = torch.cat((result, heatmaps, (batch[1] * 2 - 1).repeat(1, 3, 1, 1).detach().cpu()), dim=-2)
        #     result = torch.cat((result, heatmaps, batch[2].unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu() * 2 - 1), dim=-2)

        log['images'] = result
        
        return log

    def to_rgb(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.as_tensor(mean).to(x.device)
        std = torch.as_tensor(std).to(x.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image = x * std + mean #由image=(x-mean)/std可知，x=image*std+mean
        return image.clamp(0, 1) * 2 - 1

    # def to_rgb(self, x):    # -1~1
    #     # assert self.image_key == "segmentation"
    #     # if not hasattr(self, "colorize"):
    #     #     self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    #     # x = F.conv2d(x, weight=self.colorize)
    #     x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
    #     return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        #self.segment_act = torch.nn.Sigmoid()
        self.out_features = out_features
    def forward(self, x, encoder_features, decoder_features):
        b1, b2, b3, b4 = self.encoder_segment(x, encoder_features, decoder_features)
        output_segment = self.decoder_segment(b1, b2, b3, b4)
        if self.out_features:
            return output_segment, b1, b2, b3, b4
        else:
            return output_segment


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(4))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        
        # self.e1 = nn.Conv2d(256, base_width, 3, 1, 1)
        # self.e2 = nn.Conv2d(512, base_width * 2, 3, 1, 1)
        # self.e3 = nn.Conv2d(1024, base_width * 4, 3, 1, 1)
        # self.e4 = nn.Conv2d(2048, base_width * 8, 3, 1, 1)

        # self.d1 = nn.Conv2d(256, base_width, 3, 1, 1)
        # self.d2 = nn.Conv2d(512, base_width * 2, 3, 1, 1)
        # self.d3 = nn.Conv2d(1024, base_width * 4, 3, 1, 1)
        # self.d4 = nn.Conv2d(2048, base_width * 8, 3, 1, 1)
        
    def forward(self, x, encoder_features, decoder_features):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        # mp1 = mp1 + self.e1(encoder_features[0]) + self.d1(decoder_features[0])
        
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        # mp2 = mp2 + self.e2(encoder_features[1]) + self.d2(decoder_features[1])
        
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        # mp3 = mp3 + self.e3(encoder_features[2]) + self.d3(decoder_features[2])
        
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        # mp4 = mp4 + self.e4(encoder_features[3]) + self.d4(decoder_features[3])
        
        return mp1, mp2, mp3, mp4


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(4+4), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2+2), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * (1+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                    nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(base_width),
                                    nn.ReLU(inplace=True), 
                                    nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4):
        
        up2 = self.up2(b4)
        cat2 = torch.cat((up2,b3),dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3,b2),dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4,b1),dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out