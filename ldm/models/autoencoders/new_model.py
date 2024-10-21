import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from ldm.models.autoencoders.de_resnet import AttnBasicBlock, AttnBottleneck, de_wide_resnet50_2, BN_layer

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
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale = scale

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=1):
        super().__init__()
        self.use_conv = (upsample > 1)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if upsample > 1:
            # self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(upsample, upsample), stride=(upsample, upsample), bias=False), 
            self.upsample = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=upsample), 
                                          nn.Conv2d(in_channels, out_channels//2, 3, 1, 1, bias=False), 
                                          nn.BatchNorm2d(out_channels//2))
            self.upsample2 = nn.Conv2d(out_channels // 2 * 3, out_channels, 3, 1, 1, bias=False)

        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=1,padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        if upsample > 1:
            # self.conv2 = nn.ConvTranspose2d(out_channels//2, out_channels//2, kernel_size=upsample, stride=upsample, bias=False)
            self.conv2 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=upsample), 
                                        nn.Conv2d(out_channels//2, out_channels//2, 3, 1, 1, bias=False))
        else:
            self.conv2 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//2)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x, prevx):
                
        if self.use_conv:
            shortcut = self.upsample(x)
            shortcut = torch.cat((shortcut, prevx), dim=1)
            shortcut = self.upsample2(shortcut)
            # shortcut = shortcut + prevx
        else:
            shortcut = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        return F.relu(out + shortcut)


class ResnetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, upsample):
        super().__init__()
        
        self.num_blocks = num_blocks
        for block_id in range(num_blocks):
            block = Bottleneck(in_channels, out_channels, upsample=upsample if block_id == 0 else 1)
            in_channels = out_channels
            setattr(self, f'block_{block_id}', block)


    def forward(self, x, prevx):
        for block_id in range(self.num_blocks):
            x = getattr(self, f'block_{block_id}')(x, prevx)

        return x




class Decoder(nn.Module):
    def __init__(self, ch=64, out_ch=3, ch_mult=(32, 16, 8, 4, 1), up_samples=(2,2,2,4), num_res_blocks=(3,4,6,3),
                 dropout=0.0, resamp_with_conv=True, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        
        for unit_id in range(len(ch_mult)-2):
            unit = ResnetUnit(ch * ch_mult[unit_id], ch * ch_mult[unit_id + 1], num_res_blocks[unit_id], up_samples[unit_id])
            setattr(self, f'unit_{unit_id}', unit)
        

        # end
        # self.out = nn.Sequential(nn.ConvTranspose2d(ch * ch_mult[-2], ch * ch_mult[-1], up_samples[-1], up_samples[-1], bias=False), 
        self.out = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=up_samples[-1]), 
                                 nn.Conv2d(ch * ch_mult[-2], ch * ch_mult[-1], 1, 1, 0, bias=False), 
                                 nn.BatchNorm2d(ch * ch_mult[-1]), 
                                 nn.ReLU(), 
                                 nn.Conv2d(ch * ch_mult[-1], ch * ch_mult[-1], 3, 1, 1, bias=False), 
                                 nn.BatchNorm2d(ch * ch_mult[-1]), 
                                 nn.ReLU(), 
                                 nn.Conv2d(ch * ch_mult[-1], out_ch, 1, 1, 0, bias=False))

    def forward(self, zs):

        h = zs[-1]
        ys = []

        # upsampling
        for unit_id in range(self.num_resolutions-2):
            h = getattr(self, f'unit_{unit_id}')(h, zs[self.num_resolutions - unit_id - 3])
            ys.append(h)

        h = self.out(h)
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
        self.encoder = Backbone_ResNet(name=backbone, train_backbone=False, return_interm_layers=True, dilation=False)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        
        self.get_pos_embed = nn.ModuleList([PositionEmbeddingSine(pos_dim//2, normalize=True, temperature=10) for pos_dim in pos_dims])
        self.feature_dims = feature_dims
        embed_dims = [(fd + pd) * fold ** 2 for fd, pd, fold in zip(feature_dims, pos_dims, folds)]
        self.quantize = nn.ModuleList([VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape) for n_embed, embed_dim in zip(n_embeds, embed_dims)])
        self.shapes = [int(image_size / 2 ** i) for i in range(2, 6)]
        self.folds = folds

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        # if colorize_nlabels is not None:
        #     assert type(colorize_nlabels)==int
        #     self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
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
    
        quants = []
        diffs = []
        indices = []
        
        def cal_w(emd1, emd2):
            return torch.max(torch.cosine_similarity(emd1, emd2, dim=1).unsqueeze(1), torch.zeros(1).to(emd1.device))
                
        for i, ef in enumerate(ef_with_pe):
            qloss = 0
            efu = ef
            
            B, C, H, W = efu.shape
            for j in range(int(np.log2(self.folds[i]))):
                B, C, H, W = efu.shape
                efu = F.unfold(input=efu, kernel_size=2, stride=2, padding=0, dilation=1).view(B, -1, H // 2, W // 2)
            vq_info = self.quantize[i](efu)
            qloss += vq_info[1]
            # cos_sim = (torch.cosine_similarity(efu, vq_info[0], dim=1).unsqueeze(1) + 1) / 2
            # cos_sim = cal_w(efu, vq_info[0])
            # mid_features = cos_sim * efu + (1 - cos_sim) * vq_info[0]
            mid_features = vq_info[0]
            
            for j in range(int(np.log2(self.folds[i]))):
                H, W = mid_features.shape[2:]
                mid_features = F.fold(input=mid_features.flatten(2), output_size=(H*2, W*2), kernel_size=2, stride=2, padding=0, dilation=1)
                vq_info = self.quantize[i].get_code_scale(mid_features, (2 ** (j+1)) ** 2)
                qloss += vq_info[1]
                # cos_sim = (torch.cosine_similarity(mid_features, vq_info[0], dim=1).unsqueeze(1) + 1) / 2
                # cos_sim = cal_w(mid_features, vq_info[0])
                # mid_features = cos_sim * mid_features + (1 - cos_sim) * vq_info[0]
                mid_features = vq_info[0]
            quants.append(mid_features[:, :self.feature_dims[i], :, :])
            diffs.append(qloss)
        
        # quants.append(encoder_features[-1])

        return encoder_features, quants, diffs, indices
    
    
    def decode(self, quants):
        # quants[-1] = self.bn(quants)
        decoder_features, dec = self.decoder(quants)
        return decoder_features, dec
        # decoder_features = self.decoder(quants[-1])
        # return decoder_features, None


    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec


    def forward(self, input):
        encoder_features, quants, diffs, indices = self.encode(input)
        # quants[-1] = encoder_features[-1]
        decoder_features, dec = self.decode(quants)
        # decoder_features, dec = self.decode(encoder_features)

        return dec, diffs, encoder_features, decoder_features, quants, indices 


    def get_input(self, batch, k):
        
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format)
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()


    def training_step(self, batch, batch_idx, optimizer_idx):
        # x = self.get_input(batch, self.image_key)
        x = batch

        xrec, qloss, encoder_features, decoder_features, quants, _ = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, encoder_features, decoder_features, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss


        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, encoder_features, decoder_features, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss


    def validation_step(self, batch, batch_idx):
        xs, masks, labels = batch
                
        xrec, qloss, encoder_features, decoder_features, neighbor_features, _ = self(xs)

        
        unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)
        for i in range(3):
            B, C, H, W = encoder_features[i].shape
            encoder_features[i] = unfolder(encoder_features[i]).reshape(B, -1, H, W)
            B, C, H, W = decoder_features[i].shape
            decoder_features[i] = unfolder(decoder_features[i]).reshape(B, -1, H, W)
        
        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy()).squeeze(1)   # [B, 1, H, W]
        
        # diffs = self.loss.perceptual_loss(xs, xrec)[1]
        
        # preds = (F.interpolate(torch.sum(diffs[0], dim=1, keepdim=True), scale_factor=1, mode='bilinear').detach().cpu().numpy() + \
        #             F.interpolate(torch.sum(diffs[1], dim=1, keepdim=True), scale_factor=2, mode='bilinear').detach().cpu().numpy() + \
        #             F.interpolate(torch.sum(diffs[2], dim=1, keepdim=True), scale_factor=4, mode='bilinear').detach().cpu().numpy()).squeeze(1)   # [B, 1, H, W]
        
        
        preds = np.stack([ndimage.gaussian_filter(p, sigma=4) for p in preds], axis=0)
        # preds = np.stack(preds, axis=0)
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
                
        xrec, qloss, encoder_features, decoder_features, neighbor_features, _ = self(xs)
        
        unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)
        for i in range(3):
            B, C, H, W = encoder_features[i].shape
            encoder_features[i] = unfolder(encoder_features[i]).reshape(B, -1, H, W)
            B, C, H, W = decoder_features[i].shape
            decoder_features[i] = unfolder(decoder_features[i]).reshape(B, -1, H, W)
        
        # preds = (ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy(), sigma=8) + \
        #             ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy(), sigma=8) + \
        #             ndimage.gaussian_filter(F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy(), sigma=8)).squeeze(1)   # [B, 1, H, W]
        
        def dissimilarity(m1, m2):
            return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)
        
        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy()).squeeze(1)   # [B, 1, H, W]
        
        preds = np.stack([ndimage.gaussian_filter(p, sigma=4) for p in preds], axis=0)
        # preds = np.stack(preds, axis=0)
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
        # params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        params = list(self.decoder.parameters())
        for i in range(len(self.quantize)):
            params += list(self.quantize[i].parameters())

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    
    def get_last_layer(self):
        # return None
        return self.decoder.out[2].weight

    
    def log_images(self, batch, **kwargs):
        
        def dissimilarity(m1, m2):
            return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)
        
        def normalize(p):
            return ((p - p.min()) / (p.max() - p.min()) * 255).astype(np.uint8)
        
        log = dict()
        if kwargs['split'] == 'train':
            x = batch.to(self.device)
        else:
            x = batch[0].to(self.device)

        xrec, _, encoder_features, decoder_features, _, _ = self(x)
        # xrec = x
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
        x = self.to_rgb(x)  # [B, C, H, W]
        xrec = self.to_rgb(xrec)
        
        result = torch.cat((x, xrec), dim=-2).detach().cpu() * 2 - 1
        
        if kwargs['split'] != 'train':
            preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear') + \
                    F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear')).permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]
            _, _, h, w = xrec.shape
            heatmaps = torch.from_numpy(np.stack([cv2.applyColorMap(cv2.resize(normalize(p), (h, w)), cv2.COLORMAP_JET)[:, :, ::-1] for p in preds])).permute(0, 3, 1, 2) / 255 # [B, 3, H, W], 0~1
            heatmaps = (heatmaps * 0.3 + ((xrec.detach().cpu() + 1) / 2).cpu() * 0.5) * 2 - 1
            # result = torch.cat((result, heatmaps, (batch[1] * 2 - 1).repeat(1, 3, 1, 1).detach().cpu()), dim=-2)
            result = torch.cat((result, heatmaps, batch[1].repeat(1, 3, 1, 1).detach().cpu() * 2 - 1), dim=-2)

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
        return image.clamp(0, 1)

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
