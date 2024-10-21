import math
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange

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
from skimage import measure
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
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), down_samples=(4,2,2,2), num_res_blocks,
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

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
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
        # self.encoder = Encoder(**ddconfig)
        self.encoder = Backbone_ResNet(name=backbone, train_backbone=False, return_interm_layers=True, dilation=False)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        
        self.get_pos_embed = nn.ModuleList([PositionEmbeddingSine(pos_dim//2, normalize=True, temperature=10) for pos_dim in pos_dims])
        self.feature_dims = feature_dims
        embed_dims = [(fd + pd) * fold ** 2 for fd, pd, fold in zip(feature_dims, pos_dims, folds)]
        # embed_dims = [fd * fold ** 2 for fd, fold in zip(feature_dims, folds)]
        self.quantize = nn.ModuleList([VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape) for n_embed, embed_dim in zip(n_embeds, embed_dims)])

        self.shapes = [int(image_size / 2 ** i) for i in range(2, 6)]
        self.folds = folds

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

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
        poses = [self.get_pos_embed[i](ef).repeat(ef.shape[0], 1, 1, 1).to(ef.device) for i, ef in enumerate(encoder_features)]
        ef_with_pe = [torch.cat((ef, pos), dim=1) for ef, pos in zip(encoder_features, poses)]
        # ef_with_pe = encoder_features
        
        def cal_weight(embed1, embed2, cos_sim, window_size=1):
            cos_sim = (torch.cosine_similarity(embed1, embed2, dim=1).unsqueeze(1) + 1) / 2
            return cos_sim * (1 - np.exp(-window_size))
    
        quants = []
        diffs = []
                
        for i, ef in enumerate(ef_with_pe):
            qloss = 0
            efu = ef
            
            for j in range(int(np.log2(self.folds[i]))):
                B, C, H, W = efu.shape
                efu = F.unfold(input=efu, kernel_size=2, stride=2, padding=0, dilation=1).reshape(B, -1, H // 2, W // 2)
            vq_info = self.quantize[i](efu)
            qloss += vq_info[1]
            # cos_sim = (torch.cosine_similarity(efu, vq_info[0], dim=1).unsqueeze(1) + 1) / 2
            cos_sim = cal_weight(efu, vq_info[0], self.folds[i])
            mid_features = cos_sim * efu + (1 - cos_sim) * vq_info[0]
            
            for j in range(int(np.log2(self.folds[i]))):
                H, W = mid_features.shape[2:]
                mid_features = F.fold(input=mid_features.flatten(2), output_size=(H*2, W*2), kernel_size=2, stride=2, padding=0, dilation=1)
                vq_info = self.quantize[i].get_code_scale(mid_features, (2 ** (j+1)) ** 2)
                qloss += vq_info[1]
                # cos_sim = (torch.cosine_similarity(mid_features, vq_info[0], dim=1).unsqueeze(1) + 1) / 2
                cos_sim = cal_weight(mid_features, vq_info[0], self.folds[i])
                mid_features = cos_sim * mid_features + (1 - cos_sim) * vq_info[0]
            quants.append(mid_features[:, :self.feature_dims[i], :, :])
            diffs.append(qloss)
            

        return encoder_features, quants, diffs
    
    
    def decode(self, quants):
        decoder_features, dec = self.decoder(quants)
        return decoder_features, dec


    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec


    def forward(self, input):
        encoder_features, quants, diffs = self.encode(input)

        decoder_features, dec = self.decode(quants)

        return dec, diffs, encoder_features, decoder_features, quants 


    def get_input(self, batch, k):
        
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format)
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        # x = self.get_input(batch, self.image_key)
        # x, x_anomaly, mask, target = batch
        x = batch
        xrec, qloss, normal_x_features, decoder_features, quants = self(x)
        
        # xrec, qloss, _, decoder_features, quants = self(x_anomaly)
        # normal_x_features = self.encoder(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, normal_x_features, decoder_features, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            # preds = (F.interpolate(dissimilarity(decoder_features[0], normal_x_features[0]), scale_factor=4, mode='bilinear') + \
            #         F.interpolate(dissimilarity(decoder_features[1], normal_x_features[1]), scale_factor=8, mode='bilinear') + \
            #         F.interpolate(dissimilarity(decoder_features[2], normal_x_features[2]), scale_factor=16, mode='bilinear')).squeeze(1)   # [B, 1, H, W]
            
            # aeloss = aeloss + torch.mean(torch.abs(mask - preds))
            
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, normal_x_features, decoder_features, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
    
        xs, masks, labels = batch
        xrec, qloss, encoder_features, decoder_features, neighbor_features = self(xs)
        
        # unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)
        # for i in range(3):
        #     B, C, H, W = encoder_features[i].shape
        #     encoder_features[i] = unfolder(encoder_features[i]).reshape(B, -1, H, W)
        #     B, C, H, W = decoder_features[i].shape
        #     decoder_features[i] = unfolder(decoder_features[i]).reshape(B, -1, H, W)
        
        # preds = (ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy(), sigma=8) + \
        #             ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy(), sigma=8) + \
        #             ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy(), sigma=8)).squeeze(1)   # [B, 1, H, W]
        
        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy()).squeeze(1)   # [B, 1, H, W]
        
        preds = np.stack([ndimage.gaussian_filter(p, sigma=4) for p in preds], axis=0)
        masks[masks > 0] = 1
        res = {"preds": preds, "masks": masks.cpu().numpy(), "labels": labels.cpu().numpy()}

        return res
    
    def validation_epoch_end(self, outputs):
        preds = np.concatenate([o["preds"] for o in outputs], axis=0)
        masks = np.concatenate([o["masks"] for o in outputs], axis=0)
        labels = np.concatenate([o["labels"] for o in outputs], axis=0)
        
        preds_f = preds.flatten()
        masks_f = masks.flatten()
        
        # fpr, tpr, thresholds = metrics.roc_curve(masks, preds, pos_label=1)
        # auc_pixel = metrics.auc(fpr, tpr)
        # self.log("val/pixel_auroc", auc_pixel)
        
        
        scores = np.max(preds.reshape(preds.shape[0], -1), axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        auc_image = metrics.auc(fpr, tpr)
        self.log("val/image_auroc", auc_image)
        
        masks_1 = masks[labels > 0].squeeze(1)
        preds_1 = preds[labels > 0]
        aupros = []
        if (len(masks_1) > 0):
            for i in range(len(masks_1)):
                aupros.append(compute_pro(masks_1[i][np.newaxis, :, :], preds_1[i][np.newaxis, :, :]))
        aupro = np.mean(aupros)
        self.log("val/aupro", aupro)
        
        self.log("val/metric", aupro + auc_image)
        
        return aupro + auc_image
    
    def test_step(self, batch, batch_idx):
        xs, masks, labels = batch
        
        xrec, qloss, encoder_features, decoder_features, neighbor_features = self(xs)
        
        # unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)
        # for i in range(3):
        #     B, C, H, W = encoder_features[i].shape
        #     encoder_features[i] = unfolder(encoder_features[i]).reshape(B, -1, H, W)
        #     B, C, H, W = decoder_features[i].shape
        #     decoder_features[i] = unfolder(decoder_features[i]).reshape(B, -1, H, W)
        
        # preds = (ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy(), sigma=8) + \
        #             ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy(), sigma=8) + \
        #             ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy(), sigma=8)).squeeze(1)   # [B, 1, H, W]
        
        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy()).squeeze(1)   # [B, 1, H, W]
        
        preds = np.stack([ndimage.gaussian_filter(p, sigma=4) for p in preds], axis=0)
        masks[masks > 0] = 1
        res = {"preds": preds, "masks": masks.cpu().numpy(), "labels": labels.cpu().numpy()}

        return res
    
    def test_epoch_end(self, outputs):
        preds = np.concatenate([o["preds"] for o in outputs], axis=0)
        masks = np.concatenate([o["masks"] for o in outputs], axis=0)
        labels = np.concatenate([o["labels"] for o in outputs], axis=0)
        
        preds_f = preds.flatten()
        masks_f = masks.flatten()
        
        # fpr, tpr, thresholds = metrics.roc_curve(masks, preds, pos_label=1)
        # auc_pixel = metrics.auc(fpr, tpr)
        # self.log("val/pixel_auroc", auc_pixel)
        
        
        scores = np.max(preds.reshape(preds.shape[0], -1), axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        auc_image = metrics.auc(fpr, tpr)
        self.log("val/image_auroc", auc_image)
        
        masks_1 = masks[labels > 0].squeeze(1)
        preds_1 = preds[labels > 0]
        aupros = []
        if (len(masks_1) > 0):
            for i in range(len(masks_1)):
                aupros.append(compute_pro(masks_1[i][np.newaxis, :, :], preds_1[i][np.newaxis, :, :]))
        aupro = np.mean(aupros)
        self.log("val/aupro", aupro)
        
        self.log("val/metric", aupro + auc_image)
        
        return aupro + auc_image

    def configure_optimizers(self):
        lr = self.learning_rate
        # params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        params = list(self.decoder.parameters())
        for i in range(len(self.quantize)):
            params += list(self.quantize[i].parameters())

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
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

        xrec, _, encoder_features, decoder_features, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)  # [B, C, H, W]
            xrec = self.to_rgb(xrec)
        
        x = self.to_rgb(x)  # [B, C, H, W]
        xrec = self.to_rgb(xrec)
        
        result = torch.cat((x, xrec), dim=-2).detach().cpu()
        
        if kwargs['split'] != 'train':
            preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear')).permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]
            _, _, h, w = xrec.shape
            heatmaps = torch.from_numpy(np.stack([cv2.applyColorMap(cv2.resize(normalize(p), (h, w)), cv2.COLORMAP_JET)[:, :, ::-1] for p in preds])).permute(0, 3, 1, 2) / 255 # [B, 3, H, W], 0~1
            heatmaps = (heatmaps * 0.3 + ((xrec.detach().cpu() + 1) / 2).cpu() * 0.5) * 2 - 1
            result = torch.cat((result, heatmaps, (batch[1] * 2 - 1).repeat(1, 3, 1, 1).detach().cpu()), dim=-2)

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
    
def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, np.ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, np.ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc
