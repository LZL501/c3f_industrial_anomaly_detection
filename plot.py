import argparse
import logging
import os
import cv2

from scipy import ndimage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pprint
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.optim
import numpy as np
import yaml
from sklearn import metrics
from PIL import Image
import torchvision as tv
from matplotlib import pyplot as plt
# from eval_helper import dump, log_metrics, merge_together, performances
# from misc_helper import (
#     AverageMeter,
#     create_logger,
#     get_current_time,
#     load_state,
#     save_checkpoint,
#     set_random_seed,
#     update_config,
# )
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description="Validation Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")

from ldm.data.mvtec import MVTec_validate, MVTec_validate2
from torch.utils.data import DataLoader


def dissimilarity(m1, m2):
    return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)



def validate(path, model, device):
    model.eval()
    unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)
    
    tf = tv.transforms.Compose([tv.transforms.Resize(256), 
                                tv.transforms.ToTensor()])
    
    image = Image.open(path).convert("RGB")
    image = tf(image).unsqueeze(0) * 2 - 1
    
    mask_path = path.replace("test", "ground_truth").replace(".png", "_mask.png")
    mask = Image.open(mask_path)
    mask = tf(mask).unsqueeze(0)
    mask[mask > 0] = 1
    
    with torch.no_grad():
        image = image.to(device)
        mask = mask.to(device)
        xrec, diffs, encoder_features, decoder_features, quants, _ = model(image)
        xrec = xrec.clamp(-1, 1)

        def dissimilarity(m1, m2):
            return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)
        
        def normalize(p):
            return ((p - p.min()) / (p.max() - p.min()) * 255).astype(np.uint8)

        result = torch.cat((image, xrec), dim=-2).detach().cpu()
    
        for i in range(len(encoder_features)):
            # B, C, H, W = neighbor_features[i].shape
            # neighbor_features[i] = unfolder(neighbor_features[i]).reshape(B, -1, H, W)
            B, C, H, W = encoder_features[i].shape
            encoder_features[i] = unfolder(encoder_features[i]).reshape(B, -1, H, W)
            B, C, H, W = decoder_features[i].shape
            decoder_features[i] = unfolder(decoder_features[i]).reshape(B, -1, H, W)

        preds = (F.interpolate(dissimilarity(decoder_features[0], encoder_features[0]), scale_factor=4, mode='bilinear') + \
                F.interpolate(dissimilarity(decoder_features[1], encoder_features[1]), scale_factor=8, mode='bilinear') + \
                F.interpolate(dissimilarity(decoder_features[2], encoder_features[2]), scale_factor=16, mode='bilinear')).permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]
        
        
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        _, _, h, w = xrec.shape
        
        # heatmaps = torch.from_numpy(np.stack([cv2.applyColorMap(cv2.resize(normalize(p), (h, w)), cv2.COLORMAP_JET)[:, :, ::-1] for p in preds])).permute(0, 3, 1, 2) / 255 # [B, 3, H, W], 0~1
        # heatmaps = (heatmaps * 0.3 + ((xrec.detach().cpu() + 1) / 2).cpu() * 0.5) * 2 - 1
        # heatmaps = heatmaps * 2 - 1
        # result = torch.cat((result, heatmaps, (mask * 2 - 1).repeat(1, 3, 1, 1).detach().cpu()), dim=-2)
        # result = Image.fromarray(((result.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8))
        # result.save(f"images/{preds.max()}_{label.item()}_{i}.jpg")
        # result.save(f"plots/5.jpg")
        
        i = 14
        x = Image.fromarray(cv2.resize(np.asarray(Image.open(path)), (h, w)))
        xrec = Image.fromarray(((xrec.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8))
        heatmap = Image.fromarray(cv2.applyColorMap(cv2.resize(normalize(preds.squeeze(0)), (h, w)), cv2.COLORMAP_JET)[:, :, ::-1])
        gt = Image.fromarray(cv2.resize(np.asarray(Image.open(mask_path)), (h, w)))
        x.save(f'plots/{i}/x.jpg')
        xrec.save(f'plots/{i}/xrec.jpg')
        heatmap.save(f'plots/{i}/heatmap.jpg')
        gt.save(f'plots/{i}/gt.jpg')
        
        
        
if __name__ == "__main__":
    cls = 'zipper'
    device = "cuda"
    config = OmegaConf.load(f"configs/8/mvtec_{cls}.yaml")
    
    # ckpt = torch.load("logs/16/2023-04-18T09-21-57_mvtec_transistor/checkpoints/last.ckpt")
    # path = '/data/arima/MVTec-AD/transistor/test/bent_lead/009.png'
    
    # ckpt = torch.load("logs/2023-05-02T11-42-19_mvtec_hazelnut/checkpoints/last.ckpt")
    # path = '/data/arima/MVTec-AD/hazelnut/test/hole/001.png'
    
    
    # ckpt = torch.load("logs/2023-05-01T17-31-56_mvtec_cable/checkpoints/last.ckpt")
    # path = '/data/arima/MVTec-AD/cable/test/cable_swap/001.png'
    
    # ckpt = torch.load("logs/2023-05-02T20-52-42_mvtec_metal_nut/checkpoints/last.ckpt")
    # path = '/data/arima/MVTec-AD/metal_nut/test/bent/012.png'
    
    # ckpt = torch.load("logs/2023-04-22T16-51-37_mvtec_toothbrush/checkpoints/last.ckpt")
    # path = '/data/arima/MVTec-AD/toothbrush/test/defective/013.png'
    
    ckpt = torch.load("logs/2023-05-01T10-44-24_mvtec_zipper/checkpoints/last.ckpt")
    path = '/data/arima/MVTec-AD/zipper/test/fabric_border/015.png'
    
    
    # ckpt = torch.load("logs/2023-05-04T02-40-40_mvtec_bottle/checkpoints/last.ckpt")
    # path = 'plots/bottle/1/org.png'
    
    # ckpt = torch.load("logs/16/2023-04-17T22-27-41_mvtec_grid/checkpoints/last.ckpt")
    # path = '/data/arima/MVTec-AD/grid/test/broken/001.png'
    
    model = instantiate_from_config(config.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    # path = '/data/arima/MVTec-AD/leather/test/glue/001.png'
    
    
    
    validate(path, model, device)
