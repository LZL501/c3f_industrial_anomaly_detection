import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler:
    def run(self, features):
        return features


class BaseSampler(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod # 抽象方法，在子类中必须重写
    def run(self, features, keep_num):
        pass


class GreedyCoresetSampler(BaseSampler):
    def __init__(self):
        """Greedy Coreset sampling base class."""
        super().__init__()

    def run(self, features, keep_num):
        """Subsamples features using Greedy Coreset.
        Args:
            features: [N x D]
        """
        assert isinstance(features, torch.Tensor)
        if keep_num == features.shape[0]:
            return features
        sample_indices = self._compute_greedy_coreset_indices(features, keep_num)
        features = features.cpu()[sample_indices]
        return features

    @staticmethod
    def _compute_batchwise_differences(matrix_a, matrix_b):
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features, keep_num):
        """Runs iterative greedy coreset selection.
        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)   # [N]

        coreset_indices = []
        num_coreset_samples = keep_num

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)
            coreset_select_distance = distance_matrix[ :, select_idx : select_idx + 1]
            coreset_anchor_distances = torch.cat([coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1)
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return coreset_indices


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(self, number_of_starting_points: int = 10):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__()

    def _compute_greedy_coreset_indices(self, features, keep_num):
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = keep_num

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return coreset_indices


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
