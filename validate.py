import argparse
import logging
import os

from scipy import ndimage
import scipy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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



def validate(val_loader, model, device):
    model.eval()
    upsample_4 = nn.UpsamplingBilinear2d(scale_factor=4)
    upsample_8 = nn.UpsamplingBilinear2d(scale_factor=8)
    upsample_16 = nn.UpsamplingBilinear2d(scale_factor=16)
    upsample_32 = nn.UpsamplingBilinear2d(scale_factor=32)
    unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)

    preds_0 = []
    preds_1 = []
    preds_2 = []

    preds_0_p = []
    preds_1_p = []
    preds_2_p = []

    masks = []
    labels = []

    bgs = []
    
    l1 = torch.zeros(256).cuda()
    l2 = torch.zeros(256).cuda()
    l3 = torch.zeros(256).cuda()
    l4 = torch.zeros(256).cuda()
    
    with torch.no_grad():
        for i, (image, mask, label, bg) in enumerate(val_loader):
            
            labels.append(label.cpu().numpy())
            # if i > 10:
            #     break
            # forward
            image = image.to(device)
            mask = mask.to(device)
            xrec, diffs, encoder_features, decoder_features, neighbor_features, indices = model(image)
            for idx in indices[0]:
                l1[idx] += 1
            for idx in indices[1]:
                l2[idx] += 1
            for idx in indices[2]:
                l3[idx] += 1
            for idx in indices[3]:
                l4[idx] += 1

    l1 = np.sort(l1.cpu().numpy())[::-1]
    l2 = np.sort(l2.cpu().numpy())[::-1]
    l3 = np.sort(l3.cpu().numpy())[::-1]
    l4 = np.sort(l4.cpu().numpy())[::-1]
    
    # np.save('indices/l1_no.npy', l1)
    # np.save('indices/l2_no.npy', l2)
    # np.save('indices/l3_no.npy', l3)
    # np.save('indices/l4_no.npy', l4)
    
    np.save('indices/l1_no.npy', l1)
    np.save('indices/l2_no.npy', l2)
    np.save('indices/l3_no.npy', l3)
    np.save('indices/l4_no.npy', l4)
    
    # print(torch.topk(l1, k=20)[0])
    # print(torch.topk(l2, k=20)[0])
    # print(torch.topk(l3, k=20)[0])
    # print(torch.topk(l4, k=20)[0])
    
if __name__ == "__main__":
    cls = 'cable'
    dataset = MVTec_validate2(f"/data/arima/MVTec-AD/{cls}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = "cuda"
    config = OmegaConf.load(f"configs/8/mvtec_{cls}.yaml")
    ckpt = torch.load("logs/2023-05-03T19-03-40_mvtec_cable/checkpoints/last.ckpt")
    model = instantiate_from_config(config.model).to(device)
    model.load_state_dict(ckpt["state_dict"])

    validate(dataloader, model, device)
