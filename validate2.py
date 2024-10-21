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
    scores = []

    bgs = []
    
    with torch.no_grad():
        for i, (image, mask, label, bg) in enumerate(val_loader):
            
            labels.append(label.cpu().numpy())
            # forward
            image = image.to(device)
            mask = mask.to(device)
            xrec, diffs, encoder_features, decoder_features, neighbor_features = model(image)
            
            # scores.append()
            

            # print(neighbor_features[0].shape, unfolder(neighbor_features[0]).shape)

            # print("===============================")

            # for ef in encoder_features:
            #     print(ef.shape)

            # print("===============================")

            # for df in decoder_features:
            #     print(df.shape)

            # L2 distance
            # preds.append(upsample_4(torch.mean((decoder_features[-1] - encoder_features[0]) ** 2, dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_8(torch.mean((decoder_features[-2] - encoder_features[1]) ** 2, dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_16(torch.mean((decoder_features[-3] - encoder_features[2]) ** 2, dim=1, keepdim=True)).cpu().numpy())
            
            # L1 distance
            # preds.append(upsample_4(torch.mean((torch.abs(decoder_features[-1] - encoder_features[0])), dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_8(torch.mean((torch.abs(decoder_features[-2] - encoder_features[1])), dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_16(torch.mean((torch.abs(decoder_features[-3] - encoder_features[2])), dim=1, keepdim=True)).cpu().numpy())

            # cosine similarity
            # preds_0.append(
            #              upsample_4(torch.mean(torch.abs(decoder_features[0] - encoder_features[0]), dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_8(torch.mean(torch.abs(decoder_features[1] - encoder_features[1]), dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_16(torch.mean(torch.abs(decoder_features[2] - encoder_features[2]), dim=1, keepdim=True)).cpu().numpy())

            # preds_1.append(
            #              upsample_4(torch.mean((decoder_features[0] - encoder_features[0]) ** 2, dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_8(torch.mean((decoder_features[1] - encoder_features[1]) ** 2, dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_16(torch.mean((decoder_features[2] - encoder_features[2]) ** 2, dim=1, keepdim=True)).cpu().numpy())
            
            # preds_2.append(
            #              upsample_4(torch.mean((dissimilarity(decoder_features[0], encoder_features[0])), dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_8(torch.mean((dissimilarity(decoder_features[1], encoder_features[1])), dim=1, keepdim=True)).cpu().numpy()+
            #              upsample_16(torch.mean((dissimilarity(decoder_features[2], encoder_features[2])), dim=1, keepdim=True)).cpu().numpy())

            for i in range(len(encoder_features)):
                B, C, H, W = neighbor_features[i].shape
                neighbor_features[i] = unfolder(neighbor_features[i]).reshape(B, -1, H, W)
                B, C, H, W = encoder_features[i].shape
                encoder_features[i] = unfolder(encoder_features[i]).reshape(B, -1, H, W)
                B, C, H, W = decoder_features[i].shape
                decoder_features[i] = unfolder(decoder_features[i]).reshape(B, -1, H, W)
            
            # preds_0_p.append(
            #              upsample_4(torch.mean(torch.abs(decoder_features[0] - encoder_features[0]), dim=1, keepdim=True)).cpu()+
            #              upsample_8(torch.mean(torch.abs(decoder_features[1] - encoder_features[1]), dim=1, keepdim=True)).cpu()+
            #              upsample_16(torch.mean(torch.abs(decoder_features[2] - encoder_features[2]), dim=1, keepdim=True)).cpu())

            # preds_2_p.append(
            #              upsample_4(torch.mean((decoder_features[0] - encoder_features[0]) ** 2, dim=1, keepdim=True)).cpu()+
            #              upsample_8(torch.mean((decoder_features[1] - encoder_features[1]) ** 2, dim=1, keepdim=True)).cpu()+
            #              upsample_16(torch.mean((decoder_features[2] - encoder_features[2]) ** 2, dim=1, keepdim=True)).cpu())
            
            # preds_2_p.append(
            #              ndimage.gaussian_filter(upsample_4(torch.mean((dissimilarity(decoder_features[0], encoder_features[0])), dim=1, keepdim=True)).cpu().numpy(), sigma=4)+
            #              ndimage.gaussian_filter(upsample_8(torch.mean((dissimilarity(decoder_features[1], encoder_features[1])), dim=1, keepdim=True)).cpu().numpy(), sigma=4)+
            #              ndimage.gaussian_filter(upsample_16(torch.mean((dissimilarity(decoder_features[2], encoder_features[2])), dim=1, keepdim=True)).cpu().numpy(), sigma=4))
            
            # preds_2_p.append(
            #              upsample_4(torch.from_numpy(ndimage.gaussian_filter(dissimilarity(decoder_features[0], encoder_features[0]).cpu().numpy(), sigma=8)))+
            #              upsample_8(torch.from_numpy(ndimage.gaussian_filter(dissimilarity(decoder_features[1], encoder_features[1]).cpu().numpy(), sigma=4)))+
            #              upsample_16(torch.from_numpy(ndimage.gaussian_filter(dissimilarity(decoder_features[2], encoder_features[2]).cpu().numpy(), sigma=2))))
            
            
            # scipy.stats.wasserstein_distance()
            # preds_2_p.append((
            #              upsample_4(dissimilarity(encoder_features[0], decoder_features[0])).cpu().numpy() + 
            #              upsample_8(dissimilarity(encoder_features[1], decoder_features[1])).cpu().numpy() + 
            #              upsample_16(dissimilarity(encoder_features[2], decoder_features[2])).cpu().numpy()))
            
            _, diffs, outs0, outs1 = model.loss.perceptual_loss(image, xrec)
        
            # preds_2_p.append((F.interpolate(torch.sum(diffs[0], dim=1, keepdim=True), scale_factor=1, mode='bilinear').detach().cpu().numpy() + \
            #         F.interpolate(torch.sum(diffs[1], dim=1, keepdim=True), scale_factor=2, mode='bilinear').detach().cpu().numpy() + \
            #         F.interpolate(torch.sum(diffs[2], dim=1, keepdim=True), scale_factor=4, mode='bilinear').detach().cpu().numpy() + \
            #         F.interpolate(torch.sum(diffs[3], dim=1, keepdim=True), scale_factor=8, mode='bilinear').detach().cpu().numpy() + \
            #         F.interpolate(torch.sum(diffs[4], dim=1, keepdim=True), scale_factor=16, mode='bilinear').detach().cpu().numpy()).squeeze(1))   # [B, 1, H, W]
            
            preds_2_p.append((
                    F.interpolate(dissimilarity(outs0[2], outs1[2]), scale_factor=4, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(outs0[3], outs1[3]), scale_factor=8, mode='bilinear').detach().cpu().numpy() + \
                    F.interpolate(dissimilarity(outs0[4], outs1[4]), scale_factor=16, mode='bilinear').detach().cpu().numpy()).squeeze(1))   # [B, 1, H, W]

            masks.append(mask.cpu().numpy())

            # bgs.append(bg)

    # preds_0 = np.concatenate([pred.flatten() for pred in preds_0], axis=0)
    # preds_1 = np.concatenate([pred.flatten() for pred in preds_1], axis=0)
    # preds_2 = np.concatenate([pred.flatten() for pred in preds_2], axis=0)

    # preds_0_p = np.concatenate([pred.flatten() for pred in preds_0_p], axis=0)
    # preds_1_p = np.concatenate([pred.flatten() for pred in preds_1_p], axis=0)
    # preds_2_p = np.concatenate([ndimage.gaussian_filter(p, sigma=4) for p, bg in zip(preds_2_p, bgs)], axis=0)
    preds_2_p = np.concatenate([pred for pred in preds_2_p], axis=0)
    
    
    preds_2_p_f = np.concatenate([pred.flatten() for pred in preds_2_p], axis=0)

    masks = np.concatenate([mask.flatten() for mask in masks], axis=0)
    masks[masks > 0] = 1
    
    # fpr, tpr, thresholds = metrics.roc_curve(masks, preds_0, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print(auc)

    # fpr, tpr, thresholds = metrics.roc_curve(masks, preds_1, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print(auc)

    # fpr, tpr, thresholds = metrics.roc_curve(masks, preds_2, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print(auc)

    # fpr, tpr, thresholds = metrics.roc_curve(masks, preds_0_p, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print(auc)

    # fpr, tpr, thresholds = metrics.roc_curve(masks, preds_1_p, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print(auc)

    fpr, tpr, thresholds = metrics.roc_curve(masks, preds_2_p_f, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    
    # print(scores)
    
    scores = np.max(preds_2_p.reshape(preds_2_p.shape[0], -1), axis=-1)
    # scores = np.concatenate([torch.mean(torch.topk(torch.from_numpy(pred).reshape(pred.shape[0], -1), 30, dim=-1)[0], dim=-1).numpy() for pred in preds_2_p])
    labels = np.concatenate([label.flatten() for label in labels], axis=0)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    # print(metrics.roc_auc_score(labels, scores))
    print(auc)
   

    # gather final results
        

if __name__ == "__main__":
    cls = 'cable'
    dataset = MVTec_validate2(f"/data/arima/MVTec-AD/{cls}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = "cuda"
    config = OmegaConf.load(f"configs/8/mvtec_{cls}.yaml")
    ckpt = torch.load("logs/2023-04-29T23-44-31_mvtec_cable/checkpoints/last.ckpt")
    model = instantiate_from_config(config.model).to(device)
    model.load_state_dict(ckpt["state_dict"])

    validate(dataloader, model, device)
