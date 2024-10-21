import argparse
import logging
import os

from scipy import ndimage
import scipy
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

from ldm.data.mvtec import DatasetSplit, MVTec_validate, MVTec_validate2, MVTecDataset
from torch.utils.data import DataLoader


def dissimilarity(m1, m2):
    return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)



def validate(val_loader, model, device):
    model.eval()
    unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)

    preds_2_p = []

    masks = []
    labels = []

    bgs = []

    print(len(val_loader))
    
    with torch.no_grad():
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            start = time.time()
            for _ in range(20):
                for image in data_iterator:
                    labels.append(image["is_anomaly"].numpy().tolist())
                    mask = image["mask"].numpy().tolist()
                    image = image["image"].to(device)
                    xrec, diffs, encoder_features, decoder_features, neighbor_features, indices, _ = model(image)

                    # for i in range(len(encoder_features)):
                    #     B, C, H, W = encoder_features[i].shape
                    #     encoder_features[i] = unfolder(encoder_features[i]).reshape(B, -1, H, W)
                    #     B, C, H, W = decoder_features[i].shape
                    #     decoder_features[i] = unfolder(decoder_features[i]).reshape(B, -1, H, W)
                    
                    preds_2_p.append((
                            F.interpolate(dissimilarity(encoder_features[0], decoder_features[0]), scale_factor=4, mode='bilinear').detach().cpu().numpy() + \
                            F.interpolate(dissimilarity(encoder_features[1], decoder_features[1]), scale_factor=8, mode='bilinear').detach().cpu().numpy() + \
                            F.interpolate(dissimilarity(encoder_features[2], decoder_features[2]), scale_factor=16, mode='bilinear').detach().cpu().numpy()).squeeze(1))   # [B, 1, H, W]
            end = time.time()
            print("xxxxxxxxxxxxxxxxxxxxxxxx", end - start)
        
                

        end = time.time()
    
    print(end - start)

                
    
if __name__ == "__main__":
    cls = 'bottle'
    # dataset = MVTec_validate(f"/data/arima/MVTec-AD/{cls}")
    test_dataset = MVTecDataset(
                "/data/arima/MVTec-AD",
                classname="bottle",
                resize=256,
                imagesize=256,
                split=DatasetSplit.TEST,
                seed=0,
            )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = "cuda"
    config = OmegaConf.load(f"configs/8/mvtec_{cls}.yaml")
    ckpt = torch.load("logs/2023-07-03T16-01-38_mvtec_bottle/checkpoints/last.ckpt")
    model = instantiate_from_config(config.model).to(device)
    model.load_state_dict(ckpt["state_dict"])

    validate(dataloader, model, device)
