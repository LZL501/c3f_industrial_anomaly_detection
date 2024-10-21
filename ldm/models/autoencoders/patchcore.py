"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import sampler

from .utils import Backbone_ResNet

LOGGER = logging.getLogger(__name__)


class PatchCore(nn.Module):
    def __init__(self, backbone):
        """PatchCore anomaly detection class."""
        super().__init__()
        self.backbone = Backbone_ResNet(backbone, False, True, False)
        self.backbone.eval()
        self.device = self.backbone.device

    def load(self, input_shape, patchsize=3, patchstride=1, anomaly_score_num_nn=1, featuresampler=patchcore.sampler.IdentitySampler(), nn_method=patchcore.common.FaissNN(False, 4), **kwargs):
        self.forward_modules = torch.nn.ModuleDict({})
        self.anomaly_scorer = []
        for layer in range(len(layers_to_extract_from)):
            self.anomaly_scorer.append(patchcore.common.NearestNeighbourScorer(n_nearest_neighbours=anomaly_score_num_nn[layer], nn_method=nn_method[layer]))

        self.featuresampler = featuresampler


    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self.backbone(input_image)

        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features_layer_concat = []
        for layer in range(len(features[0])):
            features_layer = [f[layer] for f in features]
            features_layer_concat.append(torch.cat(features_layer, dim=0))
        features = [self.featuresampler.run(f) for f in features_layer_concat]
        
        for f, anomaly_scorer in zip(features, self.anomaly_scorer):
            anomaly_scorer.fit(detection_features=[f])